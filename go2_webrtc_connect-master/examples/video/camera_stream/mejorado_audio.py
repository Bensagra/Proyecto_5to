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
import platform
from pathlib import Path
from collections import deque
from torchvision.transforms import functional as F

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaPlayer

import pyaudio


# =========================
# CONFIG
# =========================
CONFIDENCE_THRESHOLD = 0.55
INPUT_WINDOW_NAME = "Unitree Go2 - Person Follow + Voice"
USE_CUDA = torch.cuda.is_available()
PERSON_CLASS_ID = 1

SAVE_DIR = Path("capturas_caras")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MIN_SECONDS_BETWEEN_SAVES = 2.0

# Seguimiento
FOLLOW_ENABLED = True
INVERT_TURN = True
MIRROR_IMAGE = False

# Comandos
COMMAND_INTERVAL = 0.04
TARGET_LOST_TIMEOUT = 0.5

# Búsqueda si pierde target
ENABLE_SEARCH_WHEN_LOST = False
SEARCH_TURN_SPEED = 0.22

# Movimiento / follow
CENTER_DEAD_ZONE = 0.035
MAX_FORWARD_SPEED = 2.0
MIN_FORWARD_SPEED = 1.0
MAX_TURN_SPEED = 0.6

DESIRED_BOX_WIDTH = 0.24
STOP_BOX_WIDTH = 0.42

SMOOTHING = 0.18
SMOOTH_ERR_ALPHA = 0.75
LAST_ERR_X = 0.0
LAST_TURN_SIGN = 0

# Audio / charla
AUTO_ACTIVATE_VOICE = True
LEG_STILL_MIN_SECONDS = 1.8
LEG_MOTION_THRESHOLD = 0.012      # más bajo = exige piernas más quietas
TALK_BOX_WIDTH = 0.32             # ancho bbox para quedar a distancia de charla
VOICE_TIMEOUT_NO_PERSON = 20.0    # cortar voz si desaparece la persona

# Audio PC
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_FRAMES_PER_BUFFER = 8192

DEBUG_PRINTS = True
logging.basicConfig(level=logging.FATAL)


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
# LEG STILL DETECTOR
# =========================
class LegStillDetector:
    """
    Detecta si la persona está quieta usando SOLO la zona de piernas:
    - parte inferior del bbox
    - recorte central horizontal para evitar brazos
    """
    def __init__(self):
        self.prev_roi = None
        self.motion_history = deque(maxlen=120)
        self.last_box = None

    def reset(self):
        self.prev_roi = None
        self.motion_history.clear()
        self.last_box = None

    def _extract_legs_roi(self, frame_bgr, person_box):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = person_box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1

        # Solo zona de piernas: desde 55% a 98% del alto del bbox
        ly1 = y1 + int(bh * 0.55)
        ly2 = y1 + int(bh * 0.98)

        # Solo parte central horizontal: evita brazos lo más posible
        lx1 = x1 + int(bw * 0.22)
        lx2 = x1 + int(bw * 0.78)

        lx1 = max(0, lx1)
        ly1 = max(0, ly1)
        lx2 = min(w, lx2)
        ly2 = min(h, ly2)

        if lx2 <= lx1 or ly2 <= ly1:
            return None

        roi = frame_bgr[ly1:ly2, lx1:lx2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.resize(gray, (96, 96))
        return gray

    def update(self, frame_bgr, person_box):
        roi = self._extract_legs_roi(frame_bgr, person_box)
        if roi is None:
            self.motion_history.append((time.time(), 1.0))
            self.prev_roi = None
            return 1.0

        motion_score = 1.0

        if self.prev_roi is not None and self.prev_roi.shape == roi.shape:
            diff = cv2.absdiff(self.prev_roi, roi)
            motion_score = float(np.mean(diff)) / 255.0

        self.prev_roi = roi
        self.motion_history.append((time.time(), motion_score))
        self.last_box = person_box[:]
        return motion_score

    def is_still(self):
        if len(self.motion_history) < 8:
            return False

        now = time.time()
        recent = [(t, m) for (t, m) in self.motion_history if now - t <= LEG_STILL_MIN_SECONDS]

        if len(recent) < 6:
            return False

        duration = recent[-1][0] - recent[0][0]
        if duration < LEG_STILL_MIN_SECONDS * 0.85:
            return False

        avg_motion = float(np.mean([m for _, m in recent]))
        return avg_motion <= LEG_MOTION_THRESHOLD

    def avg_motion(self):
        if not self.motion_history:
            return 0.0
        recent_vals = [m for _, m in list(self.motion_history)[-10:]]
        return float(np.mean(recent_vals))


# =========================
# AUDIO CHAT
# =========================
class VoiceChatManager:
    def __init__(self, conn):
        self.conn = conn
        self.active = False
        self.p = None
        self.out_stream = None
        self.mic_player = None
        self.started_at = None

    def _make_speaker_output(self):
        self.p = pyaudio.PyAudio()
        self.out_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_SAMPLE_RATE,
            output=True,
            frames_per_buffer=AUDIO_FRAMES_PER_BUFFER,
        )

    async def recv_audio_stream(self, frame):
        try:
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
            if self.out_stream is not None:
                self.out_stream.write(audio_data.tobytes())
        except Exception as e:
            print(f"[VOICE] Error reproduciendo audio: {e}")

    def _create_mic_player(self):
        """
        Best effort para distintas plataformas.
        Puede requerir ajustar el source si tu mic no entra a la primera.
        """
        system = platform.system().lower()
        candidates = []

        if system == "darwin":
            candidates = [
                ("none:0", "avfoundation"),
                ("none:1", "avfoundation"),
                (":0", "avfoundation"),
                (":1", "avfoundation"),
            ]
        elif system == "linux":
            candidates = [
                ("default", "pulse"),
                ("default", "alsa"),
            ]
        elif system == "windows":
            candidates = [
                ("audio=Microphone", "dshow"),
            ]
        else:
            candidates = [("default", None)]

        last_error = None
        for source, fmt in candidates:
            try:
                player = MediaPlayer(source, format=fmt)
                if player.audio is not None:
                    print(f"[VOICE] Mic abierto con source={source} format={fmt}")
                    return player
            except Exception as e:
                last_error = e

        raise RuntimeError(f"No pude abrir el micrófono. Último error: {last_error}")

    async def start(self):
        if self.active:
            return

        try:
            # audio robot -> compu
            self._make_speaker_output()
            self.conn.audio.switchAudioChannel(True)
            self.conn.audio.add_track_callback(self.recv_audio_stream)

            # audio compu -> robot
            self.mic_player = self._create_mic_player()
            if self.mic_player.audio is not None:
                self.conn.pc.addTrack(self.mic_player.audio)

            self.active = True
            self.started_at = time.time()
            print("[VOICE] Chat de voz ACTIVADO")

        except Exception as e:
            print(f"[VOICE] Error activando voz: {e}")
            await self.stop()

    async def stop(self):
        if not self.active and self.out_stream is None and self.p is None:
            return

        try:
            self.conn.audio.switchAudioChannel(False)
        except Exception:
            pass

        try:
            if self.out_stream is not None:
                self.out_stream.stop_stream()
                self.out_stream.close()
        except Exception:
            pass

        try:
            if self.p is not None:
                self.p.terminate()
        except Exception:
            pass

        self.out_stream = None
        self.p = None
        self.mic_player = None
        self.active = False
        self.started_at = None
        print("[VOICE] Chat de voz DESACTIVADO")


# =========================
# HELPERS
# =========================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def save_face_crop(face_crop):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    filename = SAVE_DIR / f"cara_{timestamp}_{millis:03d}.jpg"
    cv2.imwrite(str(filename), face_crop)
    return filename


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

        value = score * 2.0 + (area / (w * h)) * 2.5 - center_dist * 1.2

        if value > best_value:
            best_value = value
            best_idx = i

    return best_idx, detections[best_idx]


def compute_follow_command(target_det, frame_shape, approach_for_talk=False):
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
    # GIRO
    # -------------------------
    z = 0.0
    abs_err = abs(err_x)
    dead_zone = max(CENTER_DEAD_ZONE, 0.08)

    if abs_err > dead_zone:
        norm_err = (abs_err - dead_zone) / (1.0 - dead_zone)
        norm_err = clamp(norm_err, 0.0, 1.0)
        turn_strength = norm_err ** 1.7
        z_mag = 0.10 + turn_strength * (MAX_TURN_SPEED - 0.10)

        turn_sign = -1.0 if err_x < 0 else 1.0

        if LAST_TURN_SIGN != 0 and abs_err < 0.14:
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
    desired_width = TALK_BOX_WIDTH if approach_for_talk else DESIRED_BOX_WIDTH
    distance_error = desired_width - rel_box_w
    too_close = rel_box_w >= STOP_BOX_WIDTH or rel_box_h >= 0.78

    if too_close:
        x = 0.0
        mode = "too_close_stop"
    else:
        kp_forward = 10
        raw_x = max(0.0, distance_error * kp_forward)

        if abs_err < 0.10:
            turn_penalty = 1.0
        elif abs_err < 0.22:
            turn_penalty = 0.88
        elif abs_err < 0.35:
            turn_penalty = 0.72
        else:
            turn_penalty = 0.50

        raw_x *= turn_penalty

        if distance_error > 0.12:
            x = clamp(raw_x, 1.2, MAX_FORWARD_SPEED)
            mode = "run"
        elif distance_error > 0.08:
            x = clamp(raw_x, 0.50, MAX_FORWARD_SPEED * 0.85)
            mode = "fast_follow"
        elif distance_error > 0.025:
            x = clamp(raw_x, MIN_FORWARD_SPEED, 0.65 if approach_for_talk else 0.80)
            mode = "slow_follow"
        elif distance_error > -0.015:
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


def draw_detections(frame, detections, face_boxes=None, fps=None, follow_info=None, still_legs=False, voice_on=False, robot_mode="FOLLOW", leg_motion=0.0):
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
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        output,
        f"Personas detectadas: {len(detections)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Legs still: {'YES' if still_legs else 'NO'} | motion={leg_motion:.4f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0) if still_legs else (0, 165, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Voice: {'ON' if voice_on else 'OFF'} | State: {robot_mode}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0) if voice_on else (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if follow_info:
        text = (
            f"FOLLOW x={follow_info['x']:.2f} "
            f"z={follow_info['z']:.2f} "
            f"mode={follow_info['mode']}"
        )
        cv2.putText(
            output,
            text,
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 255),
            2,
            cv2.LINE_AA
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

    def smooth(self, new_val, prev_val):
        return prev_val * SMOOTHING + new_val * (1.0 - SMOOTHING)

    async def send_move(self, x=0.0, y=0.0, z=0.0):
        x = self.smooth(x, self.prev_x)
        y = self.smooth(y, self.prev_y)
        z = self.smooth(z, self.prev_z)

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
            await asyncio.sleep(0.02)

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
    face_memory = FaceMemory(similarity_threshold=0.55)
    leg_still_detector = LegStillDetector()
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
    last_person_seen_time = 0.0
    robot_mode = "FOLLOW"

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
                still_legs = False
                leg_motion = 0.0

                if FOLLOW_ENABLED and len(detections) > 0:
                    last_person_seen_time = now
                    target_idx, target_det = choose_target(detections, frame.shape)

                    if target_det is not None:
                        leg_motion = leg_still_detector.update(frame, target_det["box"])
                        still_legs = leg_still_detector.is_still()

                        rel_box_w = (target_det["box"][2] - target_det["box"][0]) / float(frame.shape[1])

                        if voice_chat.active:
                            robot_mode = "VOICE_CHAT"
                            follower.clear_target()

                            # si se fue mucho o desapareció un rato, cortar voz
                            if rel_box_w < 0.16:
                                future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                                future.result(timeout=5)
                                robot_mode = "FOLLOW"

                        else:
                            if still_legs and rel_box_w < TALK_BOX_WIDTH:
                                robot_mode = "APPROACH_FOR_VOICE"
                                cmd = compute_follow_command(target_det, frame.shape, approach_for_talk=True)
                                follow_info = {**cmd, "target_idx": target_idx}
                                follower.update_target(follow_info)

                            elif still_legs and rel_box_w >= TALK_BOX_WIDTH:
                                robot_mode = "READY_FOR_VOICE"

                                follow_info = {
                                    "x": 0.0,
                                    "y": 0.0,
                                    "z": 0.0,
                                    "err_x": 0.0,
                                    "rel_box_w": rel_box_w,
                                    "mode": "ready_for_voice",
                                    "target_idx": target_idx,
                                }
                                follower.update_target(follow_info)

                                if AUTO_ACTIVATE_VOICE:
                                    future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
                                    future.result(timeout=3)

                                    future = asyncio.run_coroutine_threadsafe(voice_chat.start(), loop)
                                    future.result(timeout=10)

                                    robot_mode = "VOICE_CHAT"

                            else:
                                robot_mode = "FOLLOW"
                                cmd = compute_follow_command(target_det, frame.shape, approach_for_talk=False)
                                follow_info = {**cmd, "target_idx": target_idx}
                                follower.update_target(follow_info)
                else:
                    follower.clear_target()
                    leg_still_detector.reset()

                    if voice_chat.active:
                        robot_mode = "VOICE_CHAT"
                        if now - last_person_seen_time > VOICE_TIMEOUT_NO_PERSON:
                            future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                            future.result(timeout=5)
                            robot_mode = "FOLLOW"
                    else:
                        robot_mode = "FOLLOW"

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
                    still_legs=still_legs,
                    voice_on=voice_chat.active,
                    robot_mode=robot_mode,
                    leg_motion=leg_motion,
                )
                cv2.imshow(INPUT_WINDOW_NAME, output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    print("[KEY] STOP manual")
                    future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
                    future.result(timeout=3)
                elif key == ord("n"):
                    print("[KEY] Reforzando modo normal")
                    future = asyncio.run_coroutine_threadsafe(follower.ensure_normal_mode(), loop)
                    future.result(timeout=6)
                elif key == ord("v"):
                    if voice_chat.active:
                        print("[KEY] Voice OFF")
                        future = asyncio.run_coroutine_threadsafe(voice_chat.stop(), loop)
                        future.result(timeout=5)
                    else:
                        print("[KEY] Voice ON")
                        future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
                        future.result(timeout=3)
                        future = asyncio.run_coroutine_threadsafe(voice_chat.start(), loop)
                        future.result(timeout=10)

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

        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)