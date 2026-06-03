import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import unitree_webrtc_connect
from aiortc import MediaStreamTrack
from torchvision.transforms import functional as F

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


# =========================
# CONFIG
# =========================
INPUT_WINDOW_NAME = "Unitree Go2 - User Lock Follow"
PERSON_CLASS_ID = 1
USE_CUDA = torch.cuda.is_available()


@dataclass
class FollowConfig:
    confidence_threshold: float = 0.55
    command_interval: float = 0.08
    control_loop_sleep: float = 0.02
    target_lost_timeout: float = 0.8

    mirror_image: bool = False
    invert_turn: bool = True

    center_dead_zone: float = 0.06
    turn_kp: float = 1.20
    turn_kd: float = 0.14
    max_turn_speed: float = 0.90

    desired_box_width: float = 0.24
    stop_box_width: float = 0.42
    emergency_box_width: float = 0.52
    stop_box_height: float = 0.78
    emergency_box_height: float = 0.90

    forward_kp: float = 5.2
    min_forward_speed: float = 0.11
    max_forward_speed: float = 0.55
    turn_in_place_threshold: float = 0.34

    accel_limit_x: float = 1.8
    accel_limit_z: float = 2.8
    err_smoothing: float = 0.70

    send_smoothing: float = 0.35

    lock_max_misses: int = 24
    lock_score_threshold: float = 4.1

    enable_search_when_lost: bool = False
    search_turn_speed: float = 0.24

    debug_prints: bool = True


@dataclass
class StartupState:
    ready: bool = False
    failed: bool = False
    attempt: int = 0
    message: str = "Initializing"
    started_at: float = field(default_factory=time.time)


# =========================
# HELPERS
# =========================
def clamp(value, low, high):
    return max(low, min(high, value))


def slew_rate(current, target, max_delta):
    delta = clamp(target - current, -max_delta, max_delta)
    return current + delta


def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def sanitize_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(clamp(x1, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    x2 = int(clamp(x2, 1, w))
    y2 = int(clamp(y2, 1, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


def person_histogram(frame_bgr, box):
    x1, y1, x2, y2 = sanitize_box(box, frame_bgr.shape)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def histogram_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.5
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return clamp((corr + 1.0) * 0.5, 0.0, 1.0)


# =========================
# SSD DETECTOR
# =========================
class SSDPersonDetector:
    def __init__(self, confidence_threshold=0.55):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb).to(self.device)

        outputs = self.model([tensor])[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if label != PERSON_CLASS_ID:
                continue
            if score < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": float(score),
            })

        return detections


# =========================
# TARGET LOCK (NO SWITCH)
# =========================
class UserLockTracker:
    def __init__(self, cfg: FollowConfig):
        self.cfg = cfg
        self.locked_box = None
        self.locked_score = 0.0
        self.lock_hist = None
        self.missed_frames = 0
        self.last_match_score = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def reset(self):
        self.locked_box = None
        self.locked_score = 0.0
        self.lock_hist = None
        self.missed_frames = 0
        self.last_match_score = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def has_lock(self):
        return self.locked_box is not None

    def mark_missed(self):
        if not self.has_lock():
            return

        self.missed_frames += 1
        self.velocity_x *= 0.5
        self.velocity_y *= 0.5

        if self.missed_frames > self.cfg.lock_max_misses:
            self.reset()

    def _update_lock(self, det, hist):
        new_box = det["box"][:]

        if self.locked_box is not None:
            prev_cx, prev_cy = box_center(self.locked_box)
            new_cx, new_cy = box_center(new_box)
            self.velocity_x = new_cx - prev_cx
            self.velocity_y = new_cy - prev_cy
        else:
            self.velocity_x = 0.0
            self.velocity_y = 0.0

        self.locked_box = new_box
        self.locked_score = float(det["score"])
        self.missed_frames = 0

        if hist is not None and self.lock_hist is not None:
            alpha = 0.70
            self.lock_hist = self.lock_hist * alpha + hist * (1.0 - alpha)
        else:
            self.lock_hist = hist

    def _choose_initial(self, detections, frame_shape):
        h, w = frame_shape[:2]
        frame_center_x = w / 2.0

        best_idx = None
        best_val = -1e9

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            area = box_area(det["box"])
            center_x = (x1 + x2) / 2.0
            center_dist = abs(center_x - frame_center_x) / max(1.0, w)

            val = score * 2.0 + (area / (w * h)) * 2.6 - center_dist * 1.2

            if val > best_val:
                best_val = val
                best_idx = i

        if best_idx is None:
            return None, None
        return best_idx, detections[best_idx]

    def _choose_from_lock(self, detections, frame_bgr):
        if not self.has_lock():
            return None

        h, w = frame_bgr.shape[:2]
        locked_box = self.locked_box
        lock_area = box_area(locked_box)
        lock_cx, lock_cy = box_center(locked_box)

        pred_cx = lock_cx + self.velocity_x
        pred_cy = lock_cy + self.velocity_y

        best_tuple = None

        for idx, det in enumerate(detections):
            det_box = det["box"]
            det_score = det["score"]

            det_cx, det_cy = box_center(det_box)
            center_dist = np.hypot(det_cx - pred_cx, det_cy - pred_cy) / max(1.0, max(w, h))

            if center_dist > 0.70:
                continue

            iou_val = box_iou(det_box, locked_box)
            det_area = box_area(det_box)
            area_ratio = min(det_area, lock_area) / max(det_area, lock_area)

            det_hist = person_histogram(frame_bgr, det_box)
            appearance = histogram_similarity(det_hist, self.lock_hist)

            if appearance < 0.15 and iou_val < 0.05 and center_dist > 0.35:
                continue

            center_score = clamp(1.0 - center_dist, 0.0, 1.0)

            lock_score = (
                iou_val * 4.8
                + appearance * 3.4
                + center_score * 2.1
                + area_ratio * 1.3
                + det_score * 0.8
            )

            if best_tuple is None or lock_score > best_tuple[0]:
                best_tuple = (lock_score, idx, det, det_hist)

        if best_tuple is None:
            return None

        best_score, best_idx, best_det, best_hist = best_tuple
        if best_score < self.cfg.lock_score_threshold:
            return None

        return best_idx, best_det, best_hist, best_score

    def select_target(self, detections, frame_bgr):
        if not detections:
            self.mark_missed()
            return None, None, None

        if not self.has_lock():
            idx, det = self._choose_initial(detections, frame_bgr.shape)
            if det is None:
                return None, None, None

            hist = person_histogram(frame_bgr, det["box"])
            self._update_lock(det, hist)
            self.last_match_score = 0.0
            return idx, det, None

        matched = self._choose_from_lock(detections, frame_bgr)
        if matched is None:
            self.mark_missed()
            return None, None, None

        idx, det, hist, score = matched
        self._update_lock(det, hist)
        self.last_match_score = score
        return idx, det, score


# =========================
# MOVEMENT CONTROLLER
# =========================
class MotionController:
    def __init__(self, cfg: FollowConfig):
        self.cfg = cfg
        self.prev_err_x = 0.0
        self.prev_time = time.time()
        self.cmd_x = 0.0
        self.cmd_z = 0.0

    def reset(self):
        self.prev_err_x = 0.0
        self.prev_time = time.time()
        self.cmd_x = 0.0
        self.cmd_z = 0.0

    def compute_follow_command(self, target_det, frame_shape):
        now = time.time()
        dt = max(now - self.prev_time, 1e-3)
        self.prev_time = now

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = target_det["box"]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        rel_box_w = box_w / float(max(1, w))
        rel_box_h = box_h / float(max(1, h))

        center_x = (x1 + x2) / 2.0
        err_x_raw = (center_x - (w / 2.0)) / max(1.0, (w / 2.0))

        if self.cfg.mirror_image:
            err_x_raw = -err_x_raw

        err_x = self.prev_err_x * self.cfg.err_smoothing + err_x_raw * (1.0 - self.cfg.err_smoothing)
        d_err_x = (err_x - self.prev_err_x) / dt
        self.prev_err_x = err_x

        abs_err = abs(err_x)

        desired_z = 0.0
        if abs_err > self.cfg.center_dead_zone:
            desired_z = self.cfg.turn_kp * err_x + self.cfg.turn_kd * d_err_x
            desired_z = clamp(desired_z, -self.cfg.max_turn_speed, self.cfg.max_turn_speed)

        if self.cfg.invert_turn:
            desired_z = -desired_z

        emergency_close = rel_box_w >= self.cfg.emergency_box_width or rel_box_h >= self.cfg.emergency_box_height
        too_close = rel_box_w >= self.cfg.stop_box_width or rel_box_h >= self.cfg.stop_box_height

        if emergency_close:
            desired_x = 0.0
            desired_z = 0.0
            mode = "emergency_stop"
        elif too_close:
            desired_x = 0.0
            mode = "too_close_stop"
        else:
            distance_error = self.cfg.desired_box_width - rel_box_w

            if distance_error <= 0.0:
                desired_x = 0.0
                mode = "hold_distance"
            elif abs_err >= self.cfg.turn_in_place_threshold:
                desired_x = 0.0
                mode = "turn_in_place"
            else:
                raw_x = distance_error * self.cfg.forward_kp
                raw_x = clamp(raw_x, 0.0, self.cfg.max_forward_speed)

                if 0.0 < raw_x < self.cfg.min_forward_speed:
                    raw_x = self.cfg.min_forward_speed

                turn_penalty = clamp(1.0 - abs_err * 1.25, 0.35, 1.0)
                desired_x = raw_x * turn_penalty

                if desired_x < self.cfg.min_forward_speed * 0.55:
                    desired_x = 0.0

                if distance_error > 0.14:
                    mode = "run"
                elif distance_error > 0.08:
                    mode = "fast_follow"
                else:
                    mode = "slow_follow"

        self.cmd_x = slew_rate(self.cmd_x, desired_x, self.cfg.accel_limit_x * dt)
        self.cmd_z = slew_rate(self.cmd_z, desired_z, self.cfg.accel_limit_z * dt)

        self.cmd_x = clamp(self.cmd_x, 0.0, self.cfg.max_forward_speed)
        self.cmd_z = clamp(self.cmd_z, -self.cfg.max_turn_speed, self.cfg.max_turn_speed)

        return {
            "x": float(self.cmd_x),
            "y": 0.0,
            "z": float(self.cmd_z),
            "err_x": float(err_x),
            "rel_box_w": float(rel_box_w),
            "mode": mode,
        }


# =========================
# ROBOT FOLLOWER
# =========================
class RobotFollower:
    def __init__(self, conn, cfg: FollowConfig):
        self.conn = conn
        self.cfg = cfg

        self.last_cmd_time = 0.0
        self.last_seen_time = time.time()
        self.last_follow_info = None
        self.motion_ready = False
        self.lock = threading.Lock()
        self.robot_stopped = True

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0

    async def ensure_normal_mode(self):
        if self.motion_ready:
            return

        try:
            print("[ROBOT] Checking motion mode...")
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1001},
            )

            code = response["data"]["header"]["status"]["code"]
            if code == 0:
                data = json.loads(response["data"]["data"])
                current_mode = data["name"]
                print(f"[ROBOT] Current mode: {current_mode}")

                if current_mode != "normal":
                    print("[ROBOT] Switching to normal mode...")
                    await self.conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"],
                        {
                            "api_id": 1002,
                            "parameter": {"name": "normal"},
                        },
                    )
                    await asyncio.sleep(4.0)
                    print("[ROBOT] Normal mode enabled")

            self.motion_ready = True

        except Exception as exc:
            print(f"[ROBOT] Error preparing normal mode: {exc}")

    def _smooth_axis(self, new_val, prev_val):
        alpha = self.cfg.send_smoothing
        return prev_val * (1.0 - alpha) + new_val * alpha

    async def send_move(self, x=0.0, y=0.0, z=0.0):
        x = self._smooth_axis(x, self.prev_x)
        y = self._smooth_axis(y, self.prev_y)
        z = self._smooth_axis(z, self.prev_z)

        self.prev_x = x
        self.prev_y = y
        self.prev_z = z

        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                    },
                },
            )
        except Exception as exc:
            print(f"[ROBOT] Error sending move: {exc}")

    async def hard_stop(self):
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0

        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                    },
                },
            )
        except Exception as exc:
            print(f"[ROBOT] Error sending stop: {exc}")

    def update_target(self, follow_info):
        with self.lock:
            self.last_follow_info = follow_info
            self.last_seen_time = time.time()
            self.robot_stopped = False

    def clear_target(self):
        with self.lock:
            self.last_follow_info = None

    async def control_loop(self):
        await self.ensure_normal_mode()
        await self.hard_stop()

        while True:
            await asyncio.sleep(self.cfg.control_loop_sleep)

            now = time.time()
            if now - self.last_cmd_time < self.cfg.command_interval:
                continue

            with self.lock:
                info = self.last_follow_info
                last_seen = self.last_seen_time

            if info is not None:
                await self.send_move(info["x"], info["y"], info["z"])
                self.last_cmd_time = now
                self.robot_stopped = False

                if self.cfg.debug_prints:
                    print(
                        f"[FOLLOW] x={info['x']:.2f} z={info['z']:.2f} "
                        f"err_x={info['err_x']:.2f} box_w={info['rel_box_w']:.2f} "
                        f"mode={info['mode']}"
                    )
            else:
                lost_for = now - last_seen
                if lost_for > self.cfg.target_lost_timeout:
                    if self.cfg.enable_search_when_lost:
                        z = -self.cfg.search_turn_speed if self.cfg.invert_turn else self.cfg.search_turn_speed
                        await self.send_move(0.0, 0.0, z)
                        self.robot_stopped = False
                    else:
                        if not self.robot_stopped:
                            await self.hard_stop()
                            self.robot_stopped = True

                    self.last_cmd_time = now


# =========================
# VISUALIZATION
# =========================
def draw_overlay(frame, detections, lock_tracker, fps=None, follow_info=None):
    output = frame.copy()

    target_idx = None
    if follow_info is not None:
        target_idx = follow_info.get("target_idx")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]

        color = (0, 255, 0)
        if target_idx == i:
            color = (0, 165, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"Person {score:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA,
        )

    if lock_tracker.has_lock() and target_idx is None:
        lx1, ly1, lx2, ly2 = lock_tracker.locked_box
        cv2.rectangle(output, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
        cv2.putText(
            output,
            "LOCK REACQUIRE",
            (lx1, max(20, ly1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    h, w = output.shape[:2]
    cx = w // 2
    cv2.line(output, (cx, 0), (cx, h), (255, 255, 0), 1)

    if fps is not None:
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    lock_state = "LOCKED" if lock_tracker.has_lock() else "UNLOCKED"
    cv2.putText(
        output,
        f"Target lock: {lock_state}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0) if lock_tracker.has_lock() else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"Misses: {lock_tracker.missed_frames}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if follow_info is not None:
        text = (
            f"x={follow_info['x']:.2f} "
            f"z={follow_info['z']:.2f} "
            f"mode={follow_info['mode']}"
        )
        cv2.putText(
            output,
            text,
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        "Keys: Q quit | S stop | R reset lock | N normal mode",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


# =========================
# MAIN
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Go2 user-follow with persistent target lock")
    parser.add_argument("--method", choices=["ap", "sta"], default="ap", help="Connection method")
    parser.add_argument("--ip", default="", help="Robot IP for STA mode")
    parser.add_argument("--serial", default="", help="Robot serial for STA mode")
    parser.add_argument("--confidence", type=float, default=0.55, help="Detector confidence threshold")
    parser.add_argument(
        "--no-invert-turn",
        action="store_true",
        help="Disable turn inversion if your robot turns in the correct direction already",
    )
    parser.add_argument(
        "--search-when-lost",
        action="store_true",
        help="Rotate slowly when lock is lost instead of full stop",
    )
    parser.add_argument("--connect-retries", type=int, default=3, help="WebRTC connect retries")
    parser.add_argument("--retry-delay", type=float, default=2.5, help="Seconds between retries")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=45.0,
        help="Max seconds to wait before declaring startup failure",
    )
    parser.add_argument(
        "--datachannel-timeout",
        type=float,
        default=30.0,
        help="Seconds waiting for datachannel validation",
    )
    parser.add_argument(
        "--local-request-timeout",
        type=float,
        default=6.0,
        help="Seconds for local HTTP requests during SDP exchange",
    )
    parser.add_argument(
        "--remote-request-timeout",
        type=float,
        default=12.0,
        help="Seconds for remote cloud requests",
    )
    return parser.parse_args()


def build_connection(args):
    if args.method == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    kwargs = {}
    if args.ip:
        kwargs["ip"] = args.ip
    if args.serial:
        kwargs["serialNumber"] = args.serial

    if not kwargs:
        raise ValueError("For --method sta you must provide --ip or --serial")

    return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)


def main():
    args = parse_args()

    cfg = FollowConfig(confidence_threshold=args.confidence)
    if args.no_invert_turn:
        cfg.invert_turn = False
    if args.search_when_lost:
        cfg.enable_search_when_lost = True

    os.environ.setdefault("UNITREE_DATACHANNEL_TIMEOUT_SECONDS", f"{max(5.0, args.datachannel_timeout):.1f}")
    os.environ.setdefault("UNITREE_LOCAL_REQUEST_TIMEOUT_SECONDS", f"{max(1.0, args.local_request_timeout):.1f}")
    os.environ.setdefault("UNITREE_REMOTE_REQUEST_TIMEOUT_SECONDS", f"{max(2.0, args.remote_request_timeout):.1f}")

    print(f"[ENV] unitree_webrtc_connect path: {getattr(unitree_webrtc_connect, '__file__', 'unknown')}")
    print(
        "[ENV] timeouts -> "
        f"datachannel={os.getenv('UNITREE_DATACHANNEL_TIMEOUT_SECONDS')}s, "
        f"local_http={os.getenv('UNITREE_LOCAL_REQUEST_TIMEOUT_SECONDS')}s, "
        f"remote_http={os.getenv('UNITREE_REMOTE_REQUEST_TIMEOUT_SECONDS')}s"
    )

    logging.basicConfig(level=logging.FATAL)

    frame_queue = queue.Queue(maxsize=8)
    detector = SSDPersonDetector(confidence_threshold=cfg.confidence_threshold)
    lock_tracker = UserLockTracker(cfg)
    motion_controller = MotionController(cfg)

    conn = build_connection(args)
    follower = RobotFollower(conn, cfg)
    startup_state = StartupState(message="Starting async connection thread")
    startup_lock = threading.Lock()

    running = True

    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(INPUT_WINDOW_NAME, blank)
    cv2.waitKey(1)

    async def recv_camera_stream(track: MediaStreamTrack):
        while running:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put(img)

    def set_startup_state(message=None, attempt=None, ready=None, failed=None):
        with startup_lock:
            if message is not None:
                startup_state.message = str(message)
            if attempt is not None:
                startup_state.attempt = int(attempt)
            if ready is not None:
                startup_state.ready = bool(ready)
            if failed is not None:
                startup_state.failed = bool(failed)

    def get_startup_snapshot():
        with startup_lock:
            return (
                startup_state.ready,
                startup_state.failed,
                startup_state.attempt,
                startup_state.message,
                startup_state.started_at,
            )

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            retries = max(1, int(args.connect_retries))
            retry_delay = max(0.2, float(args.retry_delay))

            for attempt in range(1, retries + 1):
                set_startup_state(
                    message=f"Connecting to Go2 (attempt {attempt}/{retries})",
                    attempt=attempt,
                    ready=False,
                    failed=False,
                )
                print(f"[WEBRTC] Connecting attempt {attempt}/{retries}...")

                try:
                    await conn.connect()
                    print("[WEBRTC] Connected")

                    conn.video.add_track_callback(recv_camera_stream)
                    conn.video.switchVideoChannel(True)
                    print("[VIDEO] Stream enabled")

                    asyncio.create_task(follower.control_loop())
                    set_startup_state(message="Connected", ready=True, failed=False)
                    return True

                except BaseException as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    logging.error("WebRTC setup error (attempt %s): %s", attempt, err)
                    print(f"[ERROR] WebRTC connect attempt {attempt}/{retries} failed: {err}")

                    try:
                        await conn.disconnect()
                    except Exception:
                        pass

                    if attempt < retries:
                        set_startup_state(message=f"Retrying in {retry_delay:.1f}s...")
                        await asyncio.sleep(retry_delay)
                    else:
                        set_startup_state(
                            message=f"Could not connect after {retries} attempts. Last error: {err}",
                            ready=False,
                            failed=True,
                        )
                        return False

            return False

        started = False
        try:
            started = loop.run_until_complete(setup())
            if started:
                loop.run_forever()
        except BaseException as exc:
            err = f"{type(exc).__name__}: {exc}"
            logging.error("Async loop crashed: %s", err)
            print(f"[ERROR] Async loop crashed: {err}")
            set_startup_state(message=f"Async loop crashed: {err}", failed=True, ready=False)
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            loop.close()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    def submit_async(coro_factory, timeout=3.0):
        if not asyncio_thread.is_alive() or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro_factory(), loop)
            future.result(timeout=timeout)
        except Exception:
            pass

    prev_time = time.time()
    last_output = blank
    startup_failure_reported = False

    try:
        while True:
            startup_ready, startup_failed, startup_attempt, startup_message, startup_started_at = get_startup_snapshot()

            if not startup_ready:
                if not startup_failed and (time.time() - startup_started_at) > max(3.0, float(args.startup_timeout)):
                    startup_message = (
                        f"Startup timeout after {args.startup_timeout:.1f}s. "
                        "Check AP/STA mode, IP/serial and robot network."
                    )
                    set_startup_state(message=startup_message, failed=True, ready=False)
                    startup_failed = True

                status_frame = blank.copy()
                status_color = (0, 220, 255) if not startup_failed else (0, 0, 255)
                line2 = startup_message
                if startup_attempt > 0 and not startup_failed:
                    line2 = f"Attempt {startup_attempt}: {startup_message}"

                cv2.putText(
                    status_frame,
                    "Go2 WebRTC startup...",
                    (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    status_frame,
                    line2[:110],
                    (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    status_frame,
                    "Press Q to quit",
                    (40, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(INPUT_WINDOW_NAME, status_frame)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q"), 27):
                    break

                if startup_failed and not startup_failure_reported:
                    print(f"[ERROR] {startup_message}")
                    startup_failure_reported = True
                continue

            frame = None
            try:
                frame = frame_queue.get(timeout=0.02)
            except queue.Empty:
                pass

            follow_info = None

            if frame is not None:
                detections = detector.detect(frame)

                target_idx = None
                target_det = None
                lock_score = None

                target_idx, target_det, lock_score = lock_tracker.select_target(detections, frame)

                if target_det is not None:
                    cmd = motion_controller.compute_follow_command(target_det, frame.shape)
                    follow_info = {**cmd, "target_idx": target_idx, "lock_score": lock_score}
                    follower.update_target(follow_info)
                else:
                    follower.clear_target()
                    motion_controller.reset()

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                last_output = draw_overlay(
                    frame,
                    detections,
                    lock_tracker,
                    fps=fps,
                    follow_info=follow_info,
                )

            cv2.imshow(INPUT_WINDOW_NAME, last_output)

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                print("[KEY] Manual STOP")
                submit_async(lambda: follower.hard_stop(), timeout=3.0)

            elif key in (ord("r"), ord("R")):
                print("[KEY] Reset target lock")
                lock_tracker.reset()
                motion_controller.reset()
                follower.clear_target()

            elif key in (ord("n"), ord("N")):
                print("[KEY] Re-check normal mode")
                submit_async(lambda: follower.ensure_normal_mode(), timeout=6.0)

    finally:
        running = False

        submit_async(lambda: follower.hard_stop(), timeout=3.0)
        submit_async(lambda: conn.disconnect(), timeout=3.0)

        cv2.destroyAllWindows()

        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

        asyncio_thread.join(timeout=2)


if __name__ == "__main__":
    main()
