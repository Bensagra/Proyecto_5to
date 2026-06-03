#!/usr/bin/env python3
"""
Seguimiento autonomo de persona con Go2:
- Camara: detector de persona (SSD Lite de torchvision)
- LiDAR: seguridad y evitacion basica de obstaculos por rt/utlidar/voxel_map_compressed
- Voz: chat en vivo bidireccional (robot <-> PC)

Teclas:
- q: salir
- v: voice chat on/off manual
- a: auto voice on/off
- s: stop inmediato
- n: reintentar modo normal
"""

import argparse
import asyncio
import json
import platform
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from torchvision.transforms import functional as F

try:
    import pyaudio
except ImportError:
    pyaudio = None

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


PERSON_CLASS_ID = 1


@dataclass
class AppConfig:
    confidence_threshold: float = 0.55

    command_interval: float = 0.04
    target_lost_timeout: float = 0.60
    center_dead_zone: float = 0.03
    turn_gain: float = 1.8
    max_turn_speed: float = 0.9
    invert_turn: bool = True
    search_turn_speed: float = 0.22
    enable_search_when_lost: bool = True

    desired_box_width: float = 0.23
    stop_box_width: float = 0.44
    talk_box_width: float = 0.32
    max_forward_speed: float = 0.70
    min_forward_speed: float = 0.18

    smoothing_x: float = 0.10
    smoothing_z: float = 0.14

    lidar_topic: str = RTC_TOPIC["ULIDAR_ARRAY"]
    lidar_switch_topic: str = RTC_TOPIC["ULIDAR_SWITCH"]
    lidar_decoder: str = "libvoxel"
    lidar_front_angle_deg: float = 60.0
    lidar_side_extra_deg: float = 45.0
    lidar_height_min: float = -0.20
    lidar_height_max: float = 1.60
    lidar_emergency_stop_m: float = 0.45
    lidar_slow_zone_m: float = 0.90
    lidar_avoid_turn_boost: float = 0.18
    lidar_stale_timeout: float = 0.90

    audio_sample_rate: int = 48000
    audio_channels: int = 2
    audio_frames_per_buffer: int = 8192
    voice_timeout_no_person: float = 25.0

    window_name: str = "Go2 Follow + LiDAR + Voice"


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class SSDPersonDetector:
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb).to(self.device)
        outputs = self.model([tensor])[0]

        boxes = outputs["boxes"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            if label == PERSON_CLASS_ID and score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.astype(int).tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score),
                })
        return detections


def choose_target(detections, frame_shape):
    if not detections:
        return None, None

    h, w = frame_shape[:2]
    frame_center_x = w / 2.0

    best_idx = None
    best_value = -1e9

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        center_x = (x1 + x2) / 2.0
        center_dist = abs(center_x - frame_center_x) / w

        value = score * 2.0 + (area / (w * h)) * 2.7 - center_dist * 1.0

        if value > best_value:
            best_value = value
            best_idx = i

    return best_idx, detections[best_idx]


class VisionCommandBuilder:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.last_err_x = 0.0

    def build(self, target_det: Dict, frame_shape: Tuple[int, int, int]) -> Dict:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = target_det["box"]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        rel_box_w = box_w / float(w)
        rel_box_h = box_h / float(h)

        center_x = (x1 + x2) / 2.0
        err_x_raw = (center_x - (w / 2.0)) / (w / 2.0)
        err_x = self.last_err_x * 0.70 + err_x_raw * 0.30
        self.last_err_x = err_x

        abs_err = abs(err_x)
        z = 0.0
        if abs_err > self.cfg.center_dead_zone:
            z_mag = clamp(abs_err * self.cfg.turn_gain, 0.20, self.cfg.max_turn_speed)
            z = -z_mag if err_x < 0 else z_mag
        if self.cfg.invert_turn:
            z = -z

        if rel_box_w >= self.cfg.stop_box_width or rel_box_h >= 0.80:
            x = 0.0
            mode = "STOP_CLOSE"
        else:
            dist_error = self.cfg.desired_box_width - rel_box_w
            if dist_error > 0.12:
                x = self.cfg.max_forward_speed
                mode = "RUN"
            elif dist_error > 0.06:
                x = self.cfg.max_forward_speed * 0.65
                mode = "FOLLOW_FAST"
            elif dist_error > 0.02:
                x = self.cfg.max_forward_speed * 0.40
                if abs_err < 0.22:
                    x = max(x, self.cfg.min_forward_speed)
                mode = "FOLLOW_SLOW"
            else:
                x = 0.0
                mode = "HOLD_DISTANCE"

        return {
            "x": float(x),
            "y": 0.0,
            "z": float(z),
            "err_x": float(err_x),
            "rel_box_w": float(rel_box_w),
            "mode": mode,
        }


def extract_lidar_points(payload) -> Optional[np.ndarray]:
    def decode_xyz(raw) -> Optional[np.ndarray]:
        if raw is None:
            return None

        if isinstance(raw, (bytes, bytearray)):
            if len(raw) % 12 != 0:
                return None
            arr = np.frombuffer(raw, dtype=np.float32)
            if len(arr) % 3 != 0:
                return None
            return arr.reshape(-1, 3)

        arr_raw = np.asarray(raw)

        # Algunas versiones entregan uint8 crudo con triples float32 XYZ.
        if arr_raw.ndim == 1 and arr_raw.dtype == np.uint8 and len(arr_raw) % 12 == 0:
            arr = arr_raw.view(np.float32)
            if len(arr) % 3 == 0:
                return arr.reshape(-1, 3)

        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1 and len(arr) % 3 == 0:
            return arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
        return None

    if payload is None:
        return None

    if isinstance(payload, np.ndarray):
        return decode_xyz(payload)

    if isinstance(payload, list):
        try:
            return decode_xyz(payload)
        except Exception:
            return None

    if not isinstance(payload, dict):
        return None

    candidates = [payload]
    if "data" in payload:
        candidates.append(payload["data"])
        if isinstance(payload["data"], dict) and "data" in payload["data"]:
            candidates.append(payload["data"]["data"])

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        origin = obj.get("origin")
        resolution = obj.get("resolution")

        if "positions" in obj and origin is not None and resolution is not None:
            vox = decode_xyz(obj["positions"])
            if vox is not None:
                org = np.asarray(origin, dtype=np.float32).reshape(1, 3)
                return org + vox * float(resolution)

        points = obj.get("points") or obj.get("xyz") or obj.get("vertices")
        if points is not None:
            try:
                decoded = decode_xyz(points)
                if decoded is not None:
                    return decoded
            except Exception:
                continue

    return None


def analyze_lidar(points: Optional[np.ndarray], cfg: AppConfig) -> Dict:
    if points is None or len(points) == 0:
        return {
            "min_front_dist": 99.0,
            "obstacle_left": False,
            "obstacle_right": False,
            "point_count": 0,
        }

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return {
            "min_front_dist": 99.0,
            "obstacle_left": False,
            "obstacle_right": False,
            "point_count": 0,
        }

    h_mask = (pts[:, 2] > cfg.lidar_height_min) & (pts[:, 2] < cfg.lidar_height_max)
    pts = pts[h_mask]
    if len(pts) == 0:
        return {
            "min_front_dist": 99.0,
            "obstacle_left": False,
            "obstacle_right": False,
            "point_count": 0,
        }

    angles = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    dists = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

    half = cfg.lidar_front_angle_deg / 2.0
    side_half = half + cfg.lidar_side_extra_deg

    front_mask = np.abs(angles) <= half
    front_dists = dists[front_mask]
    min_front = float(np.min(front_dists)) if len(front_dists) > 0 else 99.0

    left_mask = (angles > half) & (angles <= side_half)
    right_mask = (angles < -half) & (angles >= -side_half)

    left_obs = bool(len(dists[left_mask]) > 0 and np.min(dists[left_mask]) < cfg.lidar_slow_zone_m)
    right_obs = bool(len(dists[right_mask]) > 0 and np.min(dists[right_mask]) < cfg.lidar_slow_zone_m)

    return {
        "min_front_dist": min_front,
        "obstacle_left": left_obs,
        "obstacle_right": right_obs,
        "point_count": int(len(pts)),
    }


class LiveVoiceChat:
    def __init__(self, conn: UnitreeWebRTCConnection, cfg: AppConfig):
        self.conn = conn
        self.cfg = cfg

        self.active = False
        self.output_stream = None
        self.pyaudio_instance = None
        self.input_player = None
        self.recv_callback_registered = False
        self.mic_track_added = False

    def _make_output(self):
        if pyaudio is None:
            raise RuntimeError("Falta pyaudio. Instala: pip install pyaudio")

        self.pyaudio_instance = pyaudio.PyAudio()
        self.output_stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=self.cfg.audio_channels,
            rate=self.cfg.audio_sample_rate,
            output=True,
            frames_per_buffer=self.cfg.audio_frames_per_buffer,
        )

    def _create_mic_player(self):
        system = platform.system().lower()

        if system == "darwin":
            candidates = [
                ("default", None, None),
                (":default", "avfoundation", None),
                ("none:default", "avfoundation", None),
            ]
        elif system == "linux":
            candidates = [
                ("default", "pulse", None),
                ("default", "alsa", None),
            ]
        elif system == "windows":
            candidates = [
                ("audio=Microphone", "dshow", None),
            ]
        else:
            candidates = [("default", None, None)]

        last_error = None
        for source, fmt, opts in candidates:
            try:
                player = MediaPlayer(source, format=fmt, options=opts)
                if player.audio is not None:
                    print(f"[VOICE] Microfono abierto con source={source} format={fmt}")
                    return player
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"No pude abrir microfono. Ultimo error: {last_error}")

    async def recv_audio_stream(self, frame):
        try:
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
            if self.output_stream is not None:
                self.output_stream.write(audio_data.tobytes())
        except Exception as exc:
            print(f"[VOICE] Error reproduciendo audio recibido: {exc}")

    async def start(self) -> bool:
        if self.active:
            return True

        try:
            self._make_output()

            if not self.recv_callback_registered:
                self.conn.audio.add_track_callback(self.recv_audio_stream)
                self.recv_callback_registered = True

            self.conn.audio.switchAudioChannel(True)

            if not self.mic_track_added:
                self.input_player = self._create_mic_player()
                if self.input_player.audio is not None:
                    self.conn.pc.addTrack(self.input_player.audio)
                    self.mic_track_added = True

            self.active = True
            print("[VOICE] Chat de voz ACTIVADO")
            return True

        except Exception as exc:
            print(f"[VOICE] No pude activar voz en vivo: {exc}")
            await self.stop()
            return False

    async def stop(self):
        self.active = False

        try:
            self.conn.audio.switchAudioChannel(False)
        except Exception:
            pass

        try:
            if self.output_stream is not None:
                self.output_stream.stop_stream()
                self.output_stream.close()
        except Exception:
            pass

        try:
            if self.pyaudio_instance is not None:
                self.pyaudio_instance.terminate()
        except Exception:
            pass

        self.output_stream = None
        self.pyaudio_instance = None
        print("[VOICE] Chat de voz DESACTIVADO")


class RobotController:
    def __init__(self, conn: UnitreeWebRTCConnection, cfg: AppConfig):
        self.conn = conn
        self.cfg = cfg

        self.lock = threading.Lock()
        self.motion_ready = False

        self.desired_cmd: Optional[Dict] = None
        self.last_seen_time = 0.0

        self.lidar_info = {
            "min_front_dist": 99.0,
            "obstacle_left": False,
            "obstacle_right": False,
            "point_count": 0,
        }
        self.lidar_timestamp = 0.0

        self.voice_hold = False

        self.prev_x = 0.0
        self.prev_z = 0.0
        self.last_cmd_time = 0.0

    def update_desired_cmd(self, cmd: Dict):
        with self.lock:
            self.desired_cmd = cmd
            self.last_seen_time = time.time()

    def clear_desired_cmd(self):
        with self.lock:
            self.desired_cmd = None

    def set_voice_hold(self, enabled: bool):
        with self.lock:
            self.voice_hold = enabled
            if enabled:
                self.desired_cmd = None

    def get_lidar_snapshot(self):
        with self.lock:
            info = dict(self.lidar_info)
            ts = self.lidar_timestamp
        return info, ts

    def on_lidar_message(self, message):
        points = extract_lidar_points(message)
        info = analyze_lidar(points, self.cfg)
        with self.lock:
            self.lidar_info = info
            self.lidar_timestamp = time.time()

    async def ensure_normal_mode(self):
        if self.motion_ready:
            return

        try:
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1001},
            )
            code = response["data"]["header"]["status"]["code"]

            if code == 0:
                data = json.loads(response["data"]["data"])
                current_mode = data["name"]
                print(f"[ROBOT] Modo actual: {current_mode}")

                if current_mode != "normal":
                    await self.conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"],
                        {
                            "api_id": 1002,
                            "parameter": {"name": "normal"},
                        },
                    )
                    await asyncio.sleep(4.0)
                    print("[ROBOT] Modo normal activado")

            self.motion_ready = True

        except Exception as exc:
            print(f"[ROBOT] Error preparando modo normal: {exc}")

    async def enable_lidar(self):
        try:
            await self.conn.datachannel.disableTrafficSaving(True)
        except Exception:
            pass

        try:
            self.conn.datachannel.set_decoder(decoder_type=self.cfg.lidar_decoder)
        except Exception as exc:
            print(f"[LIDAR] No pude setear decoder {self.cfg.lidar_decoder}: {exc}")

        try:
            self.conn.datachannel.pub_sub.publish_without_callback(
                self.cfg.lidar_switch_topic,
                "on",
            )
            self.conn.datachannel.pub_sub.subscribe(self.cfg.lidar_topic, self.on_lidar_message)
            print(f"[LIDAR] Suscripto a {self.cfg.lidar_topic}")
        except Exception as exc:
            print(f"[LIDAR] Error activando stream: {exc}")

    async def send_move(self, x=0.0, y=0.0, z=0.0):
        x = self.prev_x * self.cfg.smoothing_x + x * (1.0 - self.cfg.smoothing_x)
        z = self.prev_z * self.cfg.smoothing_z + z * (1.0 - self.cfg.smoothing_z)

        self.prev_x = x
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
            print(f"[ROBOT] Error enviando move: {exc}")

    async def hard_stop(self):
        self.prev_x = 0.0
        self.prev_z = 0.0
        await self.send_move(0.0, 0.0, 0.0)

    def _apply_lidar_safety(self, cmd: Dict, lidar_info: Dict, lidar_age: float) -> Dict:
        safe = dict(cmd)
        safe["lidar_age"] = lidar_age
        safe["lidar_front"] = lidar_info["min_front_dist"]

        if lidar_age > self.cfg.lidar_stale_timeout:
            safe["mode"] = safe["mode"] + "|LIDAR_STALE"
            return safe

        min_front = lidar_info["min_front_dist"]
        obs_left = lidar_info["obstacle_left"]
        obs_right = lidar_info["obstacle_right"]

        if min_front < self.cfg.lidar_emergency_stop_m:
            safe["x"] = 0.0
            safe["z"] = 0.0
            safe["mode"] = "LIDAR_EMERGENCY_STOP"
            return safe

        if min_front < self.cfg.lidar_slow_zone_m:
            scale = (min_front - self.cfg.lidar_emergency_stop_m) / (
                self.cfg.lidar_slow_zone_m - self.cfg.lidar_emergency_stop_m
            )
            scale = clamp(scale, 0.0, 1.0)
            safe["x"] = float(safe["x"] * scale)
            safe["mode"] = safe["mode"] + "|LIDAR_SLOW"

        if obs_left and not obs_right:
            safe["z"] = float(safe["z"] - self.cfg.lidar_avoid_turn_boost)
            safe["mode"] = safe["mode"] + "|AVOID_LEFT"
        elif obs_right and not obs_left:
            safe["z"] = float(safe["z"] + self.cfg.lidar_avoid_turn_boost)
            safe["mode"] = safe["mode"] + "|AVOID_RIGHT"

        safe["z"] = float(clamp(safe["z"], -self.cfg.max_turn_speed, self.cfg.max_turn_speed))
        safe["x"] = float(clamp(safe["x"], 0.0, self.cfg.max_forward_speed))
        return safe

    async def control_loop(self):
        await self.ensure_normal_mode()
        await self.enable_lidar()
        await self.hard_stop()

        while True:
            await asyncio.sleep(0.01)

            now = time.time()
            if now - self.last_cmd_time < self.cfg.command_interval:
                continue

            with self.lock:
                cmd = dict(self.desired_cmd) if self.desired_cmd is not None else None
                last_seen = self.last_seen_time
                voice_hold = self.voice_hold
                lidar_info = dict(self.lidar_info)
                lidar_ts = self.lidar_timestamp

            if voice_hold:
                await self.hard_stop()
                self.last_cmd_time = now
                continue

            if cmd is not None:
                safe_cmd = self._apply_lidar_safety(cmd, lidar_info, now - lidar_ts)
                await self.send_move(safe_cmd["x"], safe_cmd["y"], safe_cmd["z"])
                self.last_cmd_time = now
            else:
                lost_for = now - last_seen
                if lost_for > self.cfg.target_lost_timeout:
                    if self.cfg.enable_search_when_lost:
                        z = -self.cfg.search_turn_speed if self.cfg.invert_turn else self.cfg.search_turn_speed
                        await self.send_move(0.0, 0.0, z)
                    else:
                        await self.hard_stop()
                    self.last_cmd_time = now


def draw_overlay(
    frame,
    detections,
    follow_info,
    fps,
    mode_text,
    voice_on,
    auto_voice,
    lidar_info,
):
    out = frame.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        color = (0, 255, 0)
        if follow_info and follow_info.get("target_idx") == i:
            color = (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"Person {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    h, w = out.shape[:2]
    cx = w // 2
    cv2.line(out, (cx, 0), (cx, h), (255, 255, 0), 1)

    cv2.putText(out, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"STATE: {mode_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"VOICE: {'ON' if voice_on else 'OFF'}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if voice_on else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"AUTO VOICE: {'ON' if auto_voice else 'OFF'}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if auto_voice else (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"LiDAR front: {lidar_info['min_front_dist']:.2f} m", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    if follow_info:
        txt = (
            f"x={follow_info['x']:.2f} z={follow_info['z']:.2f} "
            f"boxW={follow_info['rel_box_w']:.2f} mode={follow_info['mode']}"
        )
        cv2.putText(out, txt, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    cv2.putText(
        out,
        "Keys: q quit | v voice | a auto voice | s stop | n normal mode",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def build_connection(args) -> UnitreeWebRTCConnection:
    mode = args.mode.lower()

    if mode == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    if mode == "sta":
        if args.ip:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        if args.serial:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial)
        raise ValueError("En modo sta debes pasar --ip o --serial")

    if mode == "remote":
        if not args.serial or not args.username or not args.password:
            raise ValueError("En modo remote debes pasar --serial --username --password")
        return UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )

    raise ValueError(f"Modo no soportado: {args.mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Go2 seguimiento autonomo + LiDAR + chat de voz")
    parser.add_argument("--mode", choices=["ap", "sta", "remote"], default="ap")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)

    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--auto-voice", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AppConfig(confidence_threshold=args.confidence)

    frame_queue = queue.Queue(maxsize=8)

    conn = build_connection(args)
    detector = SSDPersonDetector(confidence_threshold=cfg.confidence_threshold)
    vision_builder = VisionCommandBuilder(cfg)
    controller = RobotController(conn, cfg)
    voice_chat = LiveVoiceChat(conn, cfg)

    auto_voice = args.auto_voice
    mode_text = "SEARCH"

    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(cfg.window_name, blank)
    cv2.waitKey(1)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                print("[WEBRTC] Conectado")

                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
                print("[VIDEO] Stream activado")

                asyncio.create_task(controller.control_loop())

            except Exception as exc:
                print(f"[ERROR] WebRTC setup: {exc}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    prev_time = time.time()
    last_person_seen_time = 0.0

    try:
        while True:
            if frame_queue.empty():
                time.sleep(0.005)
                continue

            frame = frame_queue.get()
            now = time.time()

            detections = detector.detect(frame)
            follow_info = None
            target_idx = None
            target_det = None

            if detections:
                target_idx, target_det = choose_target(detections, frame.shape)

            if target_det is not None:
                cmd = vision_builder.build(target_det, frame.shape)
                follow_info = {**cmd, "target_idx": target_idx}
                controller.update_desired_cmd(follow_info)
                last_person_seen_time = now

                if voice_chat.active:
                    mode_text = "VOICE_CHAT"
                else:
                    mode_text = "FOLLOW"
            else:
                controller.clear_desired_cmd()
                if voice_chat.active:
                    mode_text = "VOICE_CHAT"
                else:
                    mode_text = "SEARCH"

            lidar_info, lidar_ts = controller.get_lidar_snapshot()
            lidar_age = now - lidar_ts if lidar_ts > 0 else 999.0

            if auto_voice:
                if voice_chat.active:
                    controller.set_voice_hold(True)
                    mode_text = "VOICE_CHAT"
                    if now - last_person_seen_time > cfg.voice_timeout_no_person:
                        future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                        future.result(timeout=6)
                        controller.set_voice_hold(False)
                        mode_text = "SEARCH"
                else:
                    if (
                        follow_info is not None
                        and follow_info["rel_box_w"] >= cfg.talk_box_width
                        and abs(follow_info["err_x"]) < 0.12
                    ):
                        future = asyncio.run_coroutine_threadsafe(controller.hard_stop(), loop)
                        future.result(timeout=4)
                        controller.set_voice_hold(True)
                        future = asyncio.run_coroutine_threadsafe(voice_chat.start(), loop)
                        started = future.result(timeout=12)
                        if started:
                            mode_text = "VOICE_CHAT"
                        else:
                            controller.set_voice_hold(False)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            overlay = draw_overlay(
                frame=frame,
                detections=detections,
                follow_info=follow_info,
                fps=fps,
                mode_text=f"{mode_text} | LiDAR age {lidar_age:.2f}s",
                voice_on=voice_chat.active,
                auto_voice=auto_voice,
                lidar_info=lidar_info,
            )
            cv2.imshow(cfg.window_name, overlay)

            key = cv2.waitKeyEx(1)
            if key == ord("q"):
                break

            if key == ord("a"):
                auto_voice = not auto_voice
                print(f"[KEY] Auto voice {'ON' if auto_voice else 'OFF'}")
                if not auto_voice and voice_chat.active:
                    future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                    future.result(timeout=6)
                    controller.set_voice_hold(False)

            if key == ord("v"):
                future = asyncio.run_coroutine_threadsafe(controller.hard_stop(), loop)
                future.result(timeout=4)
                controller.clear_desired_cmd()

                if voice_chat.active:
                    print("[KEY] Voice OFF")
                    future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                    future.result(timeout=6)
                    controller.set_voice_hold(False)
                else:
                    print("[KEY] Voice ON")
                    controller.set_voice_hold(True)
                    future = asyncio.run_coroutine_threadsafe(voice_chat.start(), loop)
                    started = future.result(timeout=12)
                    if not started:
                        controller.set_voice_hold(False)

            if key == ord("s"):
                print("[KEY] STOP")
                controller.clear_desired_cmd()
                future = asyncio.run_coroutine_threadsafe(controller.hard_stop(), loop)
                future.result(timeout=4)

            if key == ord("n"):
                print("[KEY] ensure normal mode")
                future = asyncio.run_coroutine_threadsafe(controller.ensure_normal_mode(), loop)
                future.result(timeout=8)

    finally:
        try:
            future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
            future.result(timeout=6)
        except Exception:
            pass

        try:
            controller.set_voice_hold(False)
            future = asyncio.run_coroutine_threadsafe(controller.hard_stop(), loop)
            future.result(timeout=4)
        except Exception:
            pass

        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario")