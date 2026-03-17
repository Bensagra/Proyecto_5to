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
from pathlib import Path
from torchvision.transforms import functional as F

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack


# =========================
# CONFIG
# =========================
CONFIDENCE_THRESHOLD = 0.55
INPUT_WINDOW_NAME = "Unitree Go2 - Person Follow"
USE_CUDA = torch.cuda.is_available()
PERSON_CLASS_ID = 1

SAVE_DIR = Path("capturas_caras")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MIN_SECONDS_BETWEEN_SAVES = 2.0

# Seguimiento
FOLLOW_ENABLED = True

# IMPORTANTE:
# Si vuelve a girar al revés, cambiá esto a True.
INVERT_TURN = True

# Si la imagen viene espejada
MIRROR_IMAGE = False

# Comandos más frecuentes
COMMAND_INTERVAL = 0.08
TARGET_LOST_TIMEOUT = 0.5

# Control

# Distancia aproximada por tamaño del bbox
DESIRED_BOX_WIDTH = 0.20
STOP_BOX_WIDTH = 0.48

# búsqueda si pierde target
ENABLE_SEARCH_WHEN_LOST = False
SEARCH_TURN_SPEED = 0.22

# suavizado
INVERT_TURN = True

COMMAND_INTERVAL = 0.04

MAX_FORWARD_SPEED = 2
MIN_FORWARD_SPEED = 1
DESIRED_BOX_WIDTH = 0.20
STOP_BOX_WIDTH = 0.34
MAX_TURN_SPEED = 0.95
CENTER_DEAD_ZONE = 0.03

SMOOTHING = 0.18
# Suavizado de dirección
SMOOTH_ERR_ALPHA = 0.75   # más alto = más estable
LAST_ERR_X = 0.0
LAST_TURN_SIGN = 0
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

    # -------------------------
    # SUAVIZADO DEL ERROR
    # -------------------------
    err_x = LAST_ERR_X * SMOOTH_ERR_ALPHA + err_x_raw * (1.0 - SMOOTH_ERR_ALPHA)
    LAST_ERR_X = err_x

    rel_box_w = box_w / float(w)
    rel_box_h = box_h / float(h)

    # -------------------------
    # GIRO MÁS ESTABLE
    # -------------------------
    z = 0.0
    abs_err = abs(err_x)

    # zona muerta más grande para no "serpentear"
    dead_zone = max(CENTER_DEAD_ZONE, 0.08)

    if abs_err > dead_zone:
        # error normalizado fuera de la zona muerta
        norm_err = (abs_err - dead_zone) / (1.0 - dead_zone)
        norm_err = clamp(norm_err, 0.0, 1.0)

        # curva suave: arranca leve y sube más al final
        turn_strength = norm_err ** 1.7

        # mínimo giro útil + máximo configurable
        z_mag = 0.10 + turn_strength * (MAX_TURN_SPEED - 0.10)

        turn_sign = -1.0 if err_x < 0 else 1.0

        # histéresis: si estaba girando en un sentido, no cambies por micro-ruido
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
    distance_error = DESIRED_BOX_WIDTH - rel_box_w
    too_close = rel_box_w >= STOP_BOX_WIDTH or rel_box_h >= 0.78

    if too_close:
        x = 0.0
        mode = "too_close_stop"
    else:
        # velocidad proporcional a cuán lejos está
        kp_forward = 10
        raw_x = max(0.0, distance_error * kp_forward)

        # penalización por giro, pero más suave que antes
        # que no mate el avance todo el tiempo
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
            x = clamp(raw_x, MIN_FORWARD_SPEED, 0.65)
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
def draw_detections(frame, detections, face_boxes=None, fps=None, follow_info=None):
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

    if follow_info:
        text = (
            f"FOLLOW x={follow_info['x']:.2f} "
            f"z={follow_info['z']:.2f} "
            f"mode={follow_info['mode']}"
        )
        cv2.putText(
            output,
            text,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
    frame_queue = queue.Queue(maxsize=10)

    # Elegí una conexión
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000P7I9GF8A")
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    detector = SSDPersonDetector(confidence_threshold=CONFIDENCE_THRESHOLD)
    face_cropper = FaceCropper()
    follower = RobotFollower(conn)

    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(INPUT_WINDOW_NAME, blank)
    cv2.waitKey(1)

    last_save_time = 0.0

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

                if FOLLOW_ENABLED and len(detections) > 0:
                    target_idx, target_det = choose_target(detections, frame.shape)

                    if target_det is not None:
                        cmd = compute_follow_command(target_det, frame.shape)
                        follow_info = {**cmd, "target_idx": target_idx}
                        follower.update_target(follow_info)
                else:
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
                    follow_info=follow_info
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

            else:
                time.sleep(0.005)

    finally:
        try:
            future = asyncio.run_coroutine_threadsafe(follower.hard_stop(), loop)
            future.result(timeout=3)
        except Exception:
            pass

        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)


if __name__ == "__main__":
    main()