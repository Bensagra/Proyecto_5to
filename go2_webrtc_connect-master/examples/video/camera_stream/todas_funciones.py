import cv2
import time
import torch
import queue
import asyncio
import logging
import threading
import numpy as np
import torchvision
import json
import sys
from pathlib import Path
from fractions import Fraction
from torchvision.transforms import functional as F

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from av import AudioFrame
import pyaudio


# =========================
# CONFIG
# =========================
CONFIDENCE_THRESHOLD = 0.55
INPUT_WINDOW_NAME = "Unitree Go2 - Follow + Manual + Voice"
USE_CUDA = torch.cuda.is_available()
PERSON_CLASS_ID = 1

SAVE_DIR = Path("capturas_caras")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MIN_SECONDS_BETWEEN_SAVES = 2.0

# Seguimiento
FOLLOW_ENABLED = True
AUTONOMOUS_ENABLED = True
INVERT_TURN = True
MIRROR_IMAGE = False

# Comandos
COMMAND_INTERVAL = 0.03
TARGET_LOST_TIMEOUT = 0.55

# Búsqueda si pierde target
ENABLE_SEARCH_WHEN_LOST = False
SEARCH_TURN_SPEED = 0.40

# Movimiento follow
CENTER_DEAD_ZONE = 0.020
MAX_FORWARD_SPEED = 1.60
FAST_FORWARD_SPEED = 1.20
MEDIUM_FORWARD_SPEED = 0.72
MIN_FORWARD_SPEED = 0.28
MAX_TURN_SPEED = 1.25
TURN_GAIN = 3.60

DESIRED_BOX_WIDTH = 0.23
STOP_BOX_WIDTH = 0.38
EMERGENCY_STOP_BOX_WIDTH = 0.50
EMERGENCY_STOP_BOX_HEIGHT = 0.84

SMOOTHING_X = 0.18
SMOOTHING_Y = 0.25
SMOOTHING_Z = 0.12

SMOOTH_ERR_ALPHA = 0.60
LAST_ERR_X = 0.0
LAST_TURN_SIGN = 0

# Target lock / foco
FOCUS_LOCK_MAX_MISSES = 18
FOCUS_DISTANCE_WEIGHT = 2.8
FOCUS_IOU_WEIGHT = 5.0
FOCUS_SCORE_WEIGHT = 1.2
FOCUS_AREA_WEIGHT = 1.8

# Audio manual
VOICE_MANUAL_ONLY = True
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SAMPLES = 960   # 20 ms @ 48k
AUDIO_FORMAT = pyaudio.paInt16

DEBUG_PRINTS = True
logging.basicConfig(level=logging.FATAL)

# OpenCV keycodes comunes
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_LEFT = 2424832
KEY_RIGHT = 2555904


# =========================
# FACE MEMORY
# =========================
class FaceMemory:
    def __init__(self, similarity_threshold=0.6):
        self.embeddings = []
        self.threshold = similarity_threshold

    def get_embedding(self, face_img):
        face = cv2.resize(face_img, (64, 64))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face / 255.0
        return face.flatten()

    def is_new_face(self, face_img):
        emb = self.get_embedding(face_img)

        if len(self.embeddings) == 0:
            self.embeddings.append(emb)
            return True

        for saved_emb in self.embeddings:
            dist = np.linalg.norm(emb - saved_emb)
            if dist < self.threshold:
                return False

        self.embeddings.append(emb)
        return True


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
            if label == PERSON_CLASS_ID and score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.astype(int).tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score)
                })
        return detections


# =========================
# FACE DETECTOR
# =========================
class FaceCropper:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face_in_person(self, frame_bgr, person_box):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = person_box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None

        person_roi = frame_bgr[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None, None

        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return None, None

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        pad_x = int(fw * 0.15)
        pad_y = int(fh * 0.20)

        fx1 = max(0, fx - pad_x)
        fy1 = max(0, fy - pad_y)
        fx2 = min(person_roi.shape[1], fx + fw + pad_x)
        fy2 = min(person_roi.shape[0], fy + fh + pad_y)

        face_crop = person_roi[fy1:fy2, fx1:fx2]
        global_box = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
        return face_crop, global_box


# =========================
# AUDIO
# =========================
class MicrophoneAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, rate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, chunk_samples=AUDIO_CHUNK_SAMPLES):
        super().__init__()
        self.rate = rate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk_samples,
        )
        self.enabled = False
        self._timestamp = 0

    async def recv(self):
        data = self.stream.read(self.chunk_samples, exception_on_overflow=False)

        if not self.enabled:
            data = b"\x00" * len(data)

        samples = self.chunk_samples
        layout = "mono" if self.channels == 1 else "stereo"

        frame = AudioFrame(format="s16", layout=layout, samples=samples)
        frame.planes[0].update(data)
        frame.sample_rate = self.rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self.rate)
        self._timestamp += samples
        return frame

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass


class VoiceChatManager:
    def __init__(self, conn):
        self.conn = conn
        self.active = False
        self.output_p = None
        self.output_stream = None
        self.mic_track = None
        self.sender_attached = False
        self.recv_callback_attached = False

    def init_output(self):
        if self.output_p is None:
            self.output_p = pyaudio.PyAudio()
            self.output_stream = self.output_p.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                output=True,
                frames_per_buffer=AUDIO_CHUNK_SAMPLES,
            )

    def init_mic(self):
        if self.mic_track is None:
            self.mic_track = MicrophoneAudioTrack()

    def attach_sender_once(self):
        if not self.sender_attached:
            self.init_mic()
            self.conn.pc.addTrack(self.mic_track)
            self.sender_attached = True

    async def recv_audio_stream(self, frame):
        try:
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
            if self.output_stream is not None:
                self.output_stream.write(audio_data.tobytes())
        except Exception as e:
            print(f"[VOICE] Error reproduciendo audio: {e}")

    async def start(self):
        if self.active:
            return

        try:
            self.init_output()
            self.attach_sender_once()

            if not self.recv_callback_attached:
                self.conn.audio.add_track_callback(self.recv_audio_stream)
                self.recv_callback_attached = True

            self.mic_track.enabled = True
            self.conn.audio.switchAudioChannel(True)
            self.active = True
            print("[VOICE] Chat de voz ACTIVADO")
        except Exception as e:
            print(f"[VOICE] Error activando voz: {e}")
            await self.stop()

    async def stop(self):
        try:
            self.conn.audio.switchAudioChannel(False)
        except Exception:
            pass

        if self.mic_track is not None:
            self.mic_track.enabled = False

        self.active = False
        print("[VOICE] Chat de voz DESACTIVADO")

    def shutdown(self):
        try:
            if self.mic_track is not None:
                self.mic_track.close()
        except Exception:
            pass
        try:
            if self.output_stream is not None:
                self.output_stream.stop_stream()
                self.output_stream.close()
        except Exception:
            pass
        try:
            if self.output_p is not None:
                self.output_p.terminate()
        except Exception:
            pass


# =========================
# TARGET FOCUS / LOCK
# =========================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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


class FocusTracker:
    def __init__(self):
        self.locked_box = None
        self.locked_score = 0.0
        self.missed_frames = 0

    def reset(self):
        self.locked_box = None
        self.locked_score = 0.0
        self.missed_frames = 0

    def update(self, det):
        self.locked_box = det["box"][:]
        self.locked_score = det["score"]
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames > FOCUS_LOCK_MAX_MISSES:
            self.reset()

    def has_lock(self):
        return self.locked_box is not None


def choose_target(detections, frame_shape, focus_tracker):
    if not detections:
        return None, None

    h, w = frame_shape[:2]
    frame_center_x = w / 2.0

    # Si no hay lock, elegir el mejor general
    if not focus_tracker.has_lock():
        best_idx = None
        best_val = -1e9
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            score = det["score"]
            area = box_area(det["box"])
            center_x = (x1 + x2) / 2.0
            center_dist = abs(center_x - frame_center_x) / w
            val = score * 2.0 + (area / (w * h)) * 2.5 - center_dist * 1.2
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx, detections[best_idx]

    # Si hay lock, priorizar la misma persona
    locked_box = focus_tracker.locked_box
    locked_cx, locked_cy = box_center(locked_box)

    best_idx = None
    best_val = -1e9

    for i, det in enumerate(detections):
        det_box = det["box"]
        score = det["score"]
        det_cx, det_cy = box_center(det_box)
        area = box_area(det_box)

        iou_val = box_iou(det_box, locked_box)
        dist = np.hypot(det_cx - locked_cx, det_cy - locked_cy) / max(w, h)

        val = (
            iou_val * FOCUS_IOU_WEIGHT
            - dist * FOCUS_DISTANCE_WEIGHT
            + score * FOCUS_SCORE_WEIGHT
            + (area / (w * h)) * FOCUS_AREA_WEIGHT
        )

        if val > best_val:
            best_val = val
            best_idx = i

    # Si el mejor match es muy malo, igual mantenemos lock un rato
    if best_idx is None:
        return None, None

    best_det = detections[best_idx]
    if box_iou(best_det["box"], locked_box) < 0.02:
        det_cx, det_cy = box_center(best_det["box"])
        dist = np.hypot(det_cx - locked_cx, det_cy - locked_cy) / max(w, h)
        if dist > 0.30:
            return None, None

    return best_idx, best_det


# =========================
# FOLLOW HELPERS
# =========================
def save_face_crop(face_crop):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    filename = SAVE_DIR / f"cara_{timestamp}_{millis:03d}.jpg"
    cv2.imwrite(str(filename), face_crop)
    return filename


def compute_follow_command(target_det, frame_shape):
    global LAST_ERR_X, LAST_TURN_SIGN

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = target_det["box"]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    center_x = (x1 + x2) / 2.0
    err_x_raw = (center_x - (w / 2.0)) / (w / 2.0)

    if MIRROR_IMAGE:
        err_x_raw = -err_x_raw

    err_x = LAST_ERR_X * SMOOTH_ERR_ALPHA + err_x_raw * (1.0 - SMOOTH_ERR_ALPHA)
    LAST_ERR_X = err_x

    rel_box_w = box_w / float(w)
    rel_box_h = box_h / float(h)

    # -------------------------
    # GIRO MÁS FUERTE
    # -------------------------
    z = 0.0
    abs_err = abs(err_x)
    dead_zone = CENTER_DEAD_ZONE

    if abs_err > dead_zone:
        # giro más rápido cuando el target dobla
        norm_err = (abs_err - dead_zone) / max(1e-6, (1.0 - dead_zone))
        norm_err = clamp(norm_err, 0.0, 1.0)

        # curva más agresiva que antes
        turn_strength = norm_err ** 1.15
        z_mag = 0.18 + turn_strength * (MAX_TURN_SPEED - 0.18)

        turn_sign = -1.0 if err_x < 0 else 1.0

        # poca histéresis, para no hacerlo lento al doblar
        if LAST_TURN_SIGN != 0 and abs_err < 0.06:
            turn_sign = LAST_TURN_SIGN

        LAST_TURN_SIGN = turn_sign
        z = turn_sign * z_mag
    else:
        z = 0.0
        LAST_TURN_SIGN = 0

    if INVERT_TURN:
        z = -z

    # -------------------------
    # DISTANCIA / VELOCIDAD
    # -------------------------
    distance_error = DESIRED_BOX_WIDTH - rel_box_w
    too_close = rel_box_w >= STOP_BOX_WIDTH or rel_box_h >= 0.78
    emergency_close = rel_box_w >= EMERGENCY_STOP_BOX_WIDTH or rel_box_h >= EMERGENCY_STOP_BOX_HEIGHT

    if emergency_close:
        x = 0.0
        z = 0.0
        mode = "emergency_stop"

    elif too_close:
        x = 0.0
        mode = "too_close_stop"

    else:
        # menos castigo al avanzar mientras gira
        if abs_err < 0.10:
            turn_penalty = 1.0
        elif abs_err < 0.22:
            turn_penalty = 0.93
        elif abs_err < 0.35:
            turn_penalty = 0.82
        else:
            turn_penalty = 0.68

        kp_forward = 8.0
        raw_x = max(0.0, distance_error * kp_forward)
        raw_x *= turn_penalty

        if distance_error > 0.12:
            x = clamp(max(raw_x, FAST_FORWARD_SPEED), 0.0, MAX_FORWARD_SPEED)
            mode = "run"
        elif distance_error > 0.06:
            x = clamp(max(raw_x, MEDIUM_FORWARD_SPEED), 0.0, FAST_FORWARD_SPEED)
            mode = "fast_follow"
        elif distance_error > 0.02:
            x = clamp(raw_x, 0.0, MEDIUM_FORWARD_SPEED)
            if abs_err < 0.25:
                x = max(x, MIN_FORWARD_SPEED)
            mode = "slow_follow"
        elif distance_error > -0.01:
            x = 0.0
            mode = "hold_distance"
        else:
            x = 0.0
            mode = "close_stop"

    return {
        "x": float(x),
        "y": 0.0,
        "z": float(z),
        "err_x": float(err_x),
        "rel_box_w": float(rel_box_w),
        "mode": mode,
    }


def draw_detections(
    frame,
    detections,
    face_boxes=None,
    fps=None,
    follow_info=None,
    voice_on=False,
    robot_mode="AUTO",
    autonomous_enabled=True,
    focus_locked=False,
):
    output = frame.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]

        color = (0, 255, 0)
        if follow_info and follow_info.get("target_idx") == i:
            color = (0, 165, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"Persona {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(output, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    if face_boxes:
        for (fx1, fy1, fx2, fy2) in face_boxes:
            cv2.rectangle(output, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(
                output,
                "Cara",
                (fx1, max(20, fy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

    h, w = output.shape[:2]
    cx = w // 2
    cv2.line(output, (cx, 0), (cx, h), (255, 255, 0), 1)

    if fps is not None:
        cv2.putText(output, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(output, f"Personas detectadas: {len(detections)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"Modo: {robot_mode}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"Autonomo: {'ON' if autonomous_enabled else 'OFF'}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,255,0) if autonomous_enabled else (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(output, f"Voice: {'ON' if voice_on else 'OFF'}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,255,0) if voice_on else (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(output, f"Focus lock: {'ON' if focus_locked else 'OFF'}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,255,0) if focus_locked else (255,255,255), 2, cv2.LINE_AA)

    if follow_info:
        text = f"x={follow_info['x']:.2f} z={follow_info['z']:.2f} mode={follow_info['mode']}"
        cv2.putText(output, text, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2, cv2.LINE_AA)

    cv2.putText(
        output,
        "Teclas: A auto on/off | V voz on/off | Flechas/WASD mover | SPACE stop | R reset foco | N normal mode | Q salir",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


# =========================
# FOLLOWER
# =========================
class RobotFollower:
    def __init__(self, conn):
        self.conn = conn
        self.last_cmd_time = 0.0
        self.last_seen_time = 0.0
        self.last_follow_info = None
        self.motion_ready = False
        self.lock = threading.Lock()

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0

    async def ensure_normal_mode(self):
        if self.motion_ready:
            return

        try:
            print("[ROBOT] Consultando modo de movimiento...")
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1001}
            )

            code = response["data"]["header"]["status"]["code"]

            if code == 0:
                data = json.loads(response["data"]["data"])
                current_mode = data["name"]
                print(f"[ROBOT] Modo actual: {current_mode}")

                if current_mode != "normal":
                    print("[ROBOT] Cambiando a modo normal...")
                    await self.conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"],
                        {
                            "api_id": 1002,
                            "parameter": {"name": "normal"}
                        }
                    )
                    await asyncio.sleep(4.0)
                    print("[ROBOT] Modo normal activado.")
                else:
                    print("[ROBOT] Ya estaba en modo normal.")
            else:
                print("[ROBOT] No pude leer motion mode, sigo igual.")

            self.motion_ready = True

        except Exception as e:
            print(f"[ROBOT] Error preparando modo normal: {e}")

    def smooth_axis(self, new_val, prev_val, alpha):
        return prev_val * alpha + new_val * (1.0 - alpha)

    async def send_move(self, x=0.0, y=0.0, z=0.0):
        x = self.smooth_axis(x, self.prev_x, SMOOTHING_X)
        y = self.smooth_axis(y, self.prev_y, SMOOTHING_Y)
        z = self.smooth_axis(z, self.prev_z, SMOOTHING_Z)

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
                        "z": float(z)
                    }
                }
            )
        except Exception as e:
            print(f"[ROBOT] Error enviando move: {e}")

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
                        "z": 0.0
                    }
                }
            )
        except Exception as e:
            print(f"[ROBOT] Error enviando stop: {e}")

    def update_target(self, follow_info):
        with self.lock:
            self.last_follow_info = follow_info
            self.last_seen_time = time.time()

    def clear_target(self):
        with self.lock:
            self.last_follow_info = None

    async def control_loop(self):
        await self.ensure_normal_mode()
        await self.hard_stop()

        while True:
            await asyncio.sleep(0.01)

            now = time.time()
            if now - self.last_cmd_time < COMMAND_INTERVAL:
                continue

            with self.lock:
                info = self.last_follow_info
                last_seen = self.last_seen_time

            if info is not None:
                await self.send_move(info["x"], info["y"], info["z"])
                self.last_cmd_time = now

                if DEBUG_PRINTS:
                    print(
                        f"[FOLLOW] x={info['x']:.2f} z={info['z']:.2f} "
                        f"err_x={info['err_x']:.2f} box_w={info['rel_box_w']:.2f} "
                        f"mode={info['mode']}"
                    )
            else:
                lost_for = now - last_seen
                if lost_for > TARGET_LOST_TIMEOUT:
                    if ENABLE_SEARCH_WHEN_LOST:
                        z = -SEARCH_TURN_SPEED if INVERT_TURN else SEARCH_TURN_SPEED
                        await self.send_move(0.0, 0.0, z)
                    else:
                        await self.hard_stop()
                    self.last_cmd_time = now


# =========================
# MAIN
# =========================
def main():
    global AUTONOMOUS_ENABLED

    face_memory = FaceMemory(similarity_threshold=0.55)
    focus_tracker = FocusTracker()
    frame_queue = queue.Queue(maxsize=10)

    # Elegí una conexión
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000P7I9GF8A")
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    detector = SSDPersonDetector(confidence_threshold=CONFIDENCE_THRESHOLD)
    face_cropper = FaceCropper()
    follower = RobotFollower(conn)
    voice_chat = VoiceChatManager(conn)

    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(INPUT_WINDOW_NAME, blank)
    cv2.waitKey(1)

    last_save_time = 0.0
    robot_mode = "AUTO"

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
                print("[WEBRTC] Conectado.")

                # preparar audio sender una vez
                voice_chat.attach_sender_once()

                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
                print("[VIDEO] Stream activado.")

                if FOLLOW_ENABLED:
                    asyncio.create_task(follower.control_loop())

            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")
                print(f"[ERROR] WebRTC: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    prev_time = time.time()

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()

                detections = detector.detect(frame)
                face_boxes = []
                follow_info = None
                target_idx = None
                now = time.time()

                if detections:
                    idx, target_det = choose_target(detections, frame.shape, focus_tracker)

                    if target_det is not None:
                        focus_tracker.update(target_det)

                        if AUTONOMOUS_ENABLED and not voice_chat.active:
                            robot_mode = "AUTO"
                            target_idx = idx
                            cmd = compute_follow_command(target_det, frame.shape)
                            follow_info = {**cmd, "target_idx": target_idx}
                            follower.update_target(follow_info)
                        else:
                            robot_mode = "MANUAL" if not voice_chat.active else "VOICE_CHAT"
                            follower.clear_target()
                    else:
                        focus_tracker.mark_missed()
                        if AUTONOMOUS_ENABLED and not voice_chat.active:
                            follower.clear_target()
                else:
                    focus_tracker.mark_missed()
                    if AUTONOMOUS_ENABLED and not voice_chat.active:
                        follower.clear_target()

                for det in detections:
                    person_box = det["box"]
                    face_crop, face_box = face_cropper.detect_face_in_person(frame, person_box)

                    if face_box is not None:
                        face_boxes.append(face_box)

                    if (
                        face_crop is not None
                        and face_crop.size > 0
                        and (now - last_save_time) >= MIN_SECONDS_BETWEEN_SAVES
                    ):
                        is_new = face_memory.is_new_face(face_crop)
                        if is_new:
                            path = save_face_crop(face_crop)
                            print(f"[NEW PERSON] Cara guardada en: {path}")
                            last_save_time = now
                            break

                current_time = time.time()
                fps = 1.0 / max(current_time - prev_time, 1e-6)
                prev_time = current_time

                output = draw_detections(
                    frame,
                    detections,
                    face_boxes=face_boxes,
                    fps=fps,
                    follow_info=follow_info,
                    voice_on=voice_chat.active,
                    robot_mode=robot_mode,
                    autonomous_enabled=AUTONOMOUS_ENABLED,
                    focus_locked=focus_tracker.has_lock(),
                )
                cv2.imshow(INPUT_WINDOW_NAME, output)

                key = cv2.waitKeyEx(1)

                # salir
                if key == ord("q"):
                    break

                # toggle autónomo
                elif key == ord("a"):
                    AUTONOMOUS_ENABLED = not AUTONOMOUS_ENABLED
                    print(f"[KEY] Autonomo {'ON' if AUTONOMOUS_ENABLED else 'OFF'}")
                    future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
                    future.result(timeout=3)
                    follower.clear_target()
                    robot_mode = "AUTO" if AUTONOMOUS_ENABLED else "MANUAL"

                # toggle voz manual únicamente
                elif key == ord("v"):
                    future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
                    future.result(timeout=3)
                    follower.clear_target()

                    if voice_chat.active:
                        print("[KEY] Voice OFF")
                        future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                        future.result(timeout=5)
                        robot_mode = "MANUAL" if not AUTONOMOUS_ENABLED else "AUTO"
                    else:
                        print("[KEY] Voice ON")
                        future = asyncio.run_coroutine_threadsafe(voice_chat.start(), loop)
                        future.result(timeout=10)
                        robot_mode = "VOICE_CHAT"

                # forzar normal mode
                elif key == ord("n"):
                    print("[KEY] Reforzando modo normal")
                    future = asyncio.run_coroutine_threadsafe(follower.ensure_normal_mode(), loop)
                    future.result(timeout=6)

                # reset foco
                elif key == ord("r"):
                    print("[KEY] Reset foco")
                    focus_tracker.reset()

                # stop manual
                elif key == ord("s") or key == 32:
                    print("[KEY] STOP manual")
                    future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
                    future.result(timeout=3)

                # Manual movement only if autonomous OFF and voice OFF
                elif not AUTONOMOUS_ENABLED and not voice_chat.active:
                    robot_mode = "MANUAL"

                    manual_x = 0.0
                    manual_z = 0.0

                    # Flechas
                    if key == KEY_UP:
                        manual_x = 1.0
                    elif key == KEY_DOWN:
                        manual_x = -0.6
                    elif key == KEY_LEFT:
                        manual_z = 0.9
                    elif key == KEY_RIGHT:
                        manual_z = -0.9

                    # WASD backup
                    elif key == ord("w"):
                        manual_x = 1.0
                    elif key == ord("x"):
                        manual_x = -0.6
                    elif key == ord("a"):  # ya usado para toggle, no pisa por el elif anterior
                        pass
                    elif key == ord("j"):
                        manual_z = 0.9
                    elif key == ord("l"):
                        manual_z = -0.9

                    if manual_x != 0.0 or manual_z != 0.0:
                        future = asyncio.run_coroutine_threadsafe(
                            follower.send_move(manual_x, 0.0, manual_z), loop
                        )
                        future.result(timeout=3)

            else:
                time.sleep(0.005)

    finally:
        try:
            future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
            future.result(timeout=5)
        except Exception:
            pass

        try:
            future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
            future.result(timeout=3)
        except Exception:
            pass

        try:
            voice_chat.shutdown()
        except Exception:
            pass

        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)