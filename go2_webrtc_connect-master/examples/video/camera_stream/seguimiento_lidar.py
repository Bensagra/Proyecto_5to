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
from aiortc import MediaStreamTrack

# ============================================================
# IMPORTS DEL DRIVER
# ============================================================
# Si tu proyecto usa unitree_webrtc_connect, dejá esto:
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

# Si en tu entorno usás go2_webrtc_driver, cambiá SOLO estas líneas por:
# from go2_webrtc_driver.webrtc_driver import (
#     Go2WebRTCConnection as UnitreeWebRTCConnection,
#     WebRTCConnectionMethod,
# )
# from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD


# ============================================================
# CONFIG GENERAL
# ============================================================
CONFIDENCE_THRESHOLD = 0.55
INPUT_WINDOW_NAME = "Go2 Person Follow + LiDAR"
USE_CUDA = torch.cuda.is_available()
PERSON_CLASS_ID = 1

SAVE_DIR = Path("capturas_caras")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MIN_SECONDS_BETWEEN_SAVES = 2.0

logging.basicConfig(level=logging.FATAL)

# ============================================================
# CONFIG FOLLOW
# ============================================================
FOLLOW_ENABLED = True
INVERT_TURN = True
MIRROR_IMAGE = False

# Más agresivo para no perder a la persona
COMMAND_INTERVAL = 0.03
TARGET_LOST_TIMEOUT = 0.45

# Giro
CENTER_DEAD_ZONE = 0.025
MAX_TURN_SPEED = 1.25
TURN_GAIN = 2.4

# Suavizado del error horizontal
SMOOTH_ERR_ALPHA = 0.72
LAST_ERR_X = 0.0
LAST_TURN_SIGN = 0

# Velocidad
MAX_FORWARD_SPEED = 2.0
MIN_FORWARD_SPEED = 0.22

# Fallback visual si no hay lidar
DESIRED_BOX_WIDTH = 0.20
STOP_BOX_WIDTH = 0.34

# Filtro de comandos al robot
SMOOTH_X = 0.10
SMOOTH_Y = 0.20
SMOOTH_Z = 0.20

# ============================================================
# CONFIG LIDAR
# ============================================================
USE_LIDAR = True
LIDAR_TARGET_DISTANCE = 1.8    # distancia deseada a la persona
LIDAR_STOP_DISTANCE = 1.15     # si está más cerca, stop
LIDAR_MAX_DISTANCE = 8.0

# Ventana frontal en metros
LIDAR_FRONT_MIN_X = 0.20
LIDAR_FRONT_MAX_X = 8.0
LIDAR_SIDE_LIMIT = 0.75
LIDAR_MIN_Z = -0.30
LIDAR_MAX_Z = 1.80

latest_lidar_distance = None
latest_lidar_points = []
lidar_lock = threading.Lock()

DEBUG_PRINTS = True


# ============================================================
# HELPERS
# ============================================================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def save_face_crop(face_crop):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    filename = SAVE_DIR / f"cara_{timestamp}_{millis:03d}.jpg"
    cv2.imwrite(str(filename), face_crop)
    return filename


# ============================================================
# FACE MEMORY
# ============================================================
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


# ============================================================
# SSD DETECTOR
# ============================================================
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
                    "score": float(score),
                })

        return detections


# ============================================================
# FACE DETECTOR
# ============================================================
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
            minSize=(40, 40),
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


# ============================================================
# TARGET SELECTION
# ============================================================
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


# ============================================================
# LIDAR PARSER
# ============================================================
def extract_positions_from_lidar_message(message):
    """
    Intenta extraer posiciones desde varios formatos posibles.
    Devuelve lista de puntos [(x,y,z), ...]
    """
    candidates = []

    if isinstance(message, dict):
        candidates.append(message)
        if "data" in message:
            candidates.append(message["data"])
            if isinstance(message["data"], dict) and "data" in message["data"]:
                candidates.append(message["data"]["data"])

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        for key in ["positions", "points", "xyz", "vertices"]:
            if key not in obj:
                continue

            positions = obj[key]

            if isinstance(positions, list) and len(positions) >= 3:
                # Formato plano: [x,y,z,x,y,z,...]
                if isinstance(positions[0], (int, float)):
                    pts = []
                    for i in range(0, len(positions) - 2, 3):
                        pts.append((
                            float(positions[i]),
                            float(positions[i + 1]),
                            float(positions[i + 2]),
                        ))
                    return pts

                # Formato [[x,y,z], ...]
                if isinstance(positions[0], (list, tuple)) and len(positions[0]) >= 3:
                    pts = []
                    for p in positions:
                        pts.append((float(p[0]), float(p[1]), float(p[2])))
                    return pts

    return []


def lidar_callback(message):
    global latest_lidar_distance, latest_lidar_points

    try:
        points = extract_positions_from_lidar_message(message)
        if not points:
            return

        frontal = []
        for x, y, z in points:
            # Asumimos: x adelante, y lateral, z altura
            if (
                LIDAR_FRONT_MIN_X < x < LIDAR_FRONT_MAX_X
                and abs(y) < LIDAR_SIDE_LIMIT
                and LIDAR_MIN_Z < z < LIDAR_MAX_Z
            ):
                frontal.append((x, y, z))

        if not frontal:
            return

        xs = sorted(p[0] for p in frontal)
        idx = max(0, int(len(xs) * 0.20) - 1)
        dist = xs[idx]

        with lidar_lock:
            latest_lidar_points = frontal
            latest_lidar_distance = dist

    except Exception as e:
        print(f"[LIDAR] Error: {e}")


# ============================================================
# CONTROL
# ============================================================
def compute_follow_command(target_det, frame_shape):
    global LAST_ERR_X, LAST_TURN_SIGN, latest_lidar_distance

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = target_det["box"]

    center_x = (x1 + x2) / 2.0
    err_x_raw = (center_x - (w / 2.0)) / (w / 2.0)

    if MIRROR_IMAGE:
        err_x_raw = -err_x_raw

    err_x = LAST_ERR_X * SMOOTH_ERR_ALPHA + err_x_raw * (1.0 - SMOOTH_ERR_ALPHA)
    LAST_ERR_X = err_x

    abs_err = abs(err_x)

    # -------------------------
    # GIRO MÁS RÁPIDO
    # -------------------------
    z = 0.0
    if abs_err > CENTER_DEAD_ZONE:
        norm_err = (abs_err - CENTER_DEAD_ZONE) / (1.0 - CENTER_DEAD_ZONE)
        norm_err = clamp(norm_err, 0.0, 1.0)

        turn_strength = norm_err ** 1.15
        z_mag = 0.15 + turn_strength * (MAX_TURN_SPEED - 0.15)
        z_mag = min(MAX_TURN_SPEED, z_mag * TURN_GAIN)

        turn_sign = -1.0 if err_x < 0 else 1.0

        # pequeña histéresis para no cambiar de lado por ruido
        if LAST_TURN_SIGN != 0 and abs_err < 0.10:
            turn_sign = LAST_TURN_SIGN

        LAST_TURN_SIGN = turn_sign
        z = turn_sign * z_mag
    else:
        LAST_TURN_SIGN = 0
        z = 0.0

    if INVERT_TURN:
        z = -z

    # -------------------------
    # DISTANCIA CON LIDAR
    # -------------------------
    with lidar_lock:
        lidar_dist = latest_lidar_distance

    if USE_LIDAR and lidar_dist is not None:
        dist_error = lidar_dist - LIDAR_TARGET_DISTANCE

        if lidar_dist <= LIDAR_STOP_DISTANCE:
            x = 0.0
            mode = "lidar_stop"
        else:
            kp = 1.35
            raw_x = max(0.0, dist_error * kp)

            # penalización leve por giro, pero sin matar el avance
            if abs_err < 0.08:
                turn_factor = 1.0
            elif abs_err < 0.20:
                turn_factor = 0.88
            elif abs_err < 0.35:
                turn_factor = 0.72
            else:
                turn_factor = 0.55

            raw_x *= turn_factor

            if dist_error > 1.6:
                x = clamp(raw_x, 1.25, MAX_FORWARD_SPEED)
                mode = "run_lidar"
            elif dist_error > 0.8:
                x = clamp(raw_x, 0.70, MAX_FORWARD_SPEED * 0.85)
                mode = "fast_lidar"
            elif dist_error > 0.2:
                x = clamp(raw_x, MIN_FORWARD_SPEED, 0.65)
                mode = "slow_lidar"
            elif dist_error > -0.05:
                x = 0.0
                mode = "hold_lidar"
            else:
                x = 0.0
                mode = "close_lidar"
    else:
        # fallback visual
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        rel_box_w = box_w / float(w)
        rel_box_h = box_h / float(h)

        distance_error = DESIRED_BOX_WIDTH - rel_box_w
        too_close = rel_box_w >= STOP_BOX_WIDTH or rel_box_h >= 0.78

        if too_close:
            x = 0.0
            mode = "bbox_stop"
        else:
            kp_forward = 6.0
            raw_x = max(0.0, distance_error * kp_forward)

            if abs_err < 0.10:
                turn_factor = 1.0
            elif abs_err < 0.22:
                turn_factor = 0.88
            elif abs_err < 0.35:
                turn_factor = 0.72
            else:
                turn_factor = 0.50

            raw_x *= turn_factor

            if distance_error > 0.18:
                x = clamp(raw_x, 0.90, MAX_FORWARD_SPEED)
                mode = "run_bbox"
            elif distance_error > 0.08:
                x = clamp(raw_x, 0.50, MAX_FORWARD_SPEED * 0.85)
                mode = "fast_bbox"
            elif distance_error > 0.025:
                x = clamp(raw_x, MIN_FORWARD_SPEED, 0.65)
                mode = "slow_bbox"
            elif distance_error > -0.015:
                x = 0.0
                mode = "hold_bbox"
            else:
                x = 0.0
                mode = "close_bbox"

    return {
        "x": float(x),
        "y": 0.0,
        "z": float(z),
        "err_x": float(err_x),
        "lidar_distance": None if lidar_dist is None else float(lidar_dist),
        "mode": mode,
    }


# ============================================================
# DIBUJO
# ============================================================
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
            output, label, (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA
        )

    if face_boxes:
        for (fx1, fy1, fx2, fy2) in face_boxes:
            cv2.rectangle(output, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(
                output, "Cara", (fx1, max(20, fy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA
            )

    h, w = output.shape[:2]
    cx = w // 2
    cv2.line(output, (cx, 0), (cx, h), (255, 255, 0), 1)

    if fps is not None:
        cv2.putText(
            output, f"FPS: {fps:.1f}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA
        )

    cv2.putText(
        output, f"Personas detectadas: {len(detections)}", (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
    )

    if follow_info:
        lidar_txt = "None"
        if follow_info.get("lidar_distance") is not None:
            lidar_txt = f"{follow_info['lidar_distance']:.2f}m"

        text = (
            f"FOLLOW x={follow_info['x']:.2f} "
            f"z={follow_info['z']:.2f} "
            f"lidar={lidar_txt} "
            f"mode={follow_info['mode']}"
        )
        cv2.putText(
            output, text, (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA
        )

    return output


# ============================================================
# ROBOT FOLLOWER
# ============================================================
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
            print("[ROBOT] Consultando modo...")
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
                    print("[ROBOT] Cambiando a normal...")
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
                    print("[ROBOT] Ya estaba en normal.")

            self.motion_ready = True

        except Exception as e:
            print(f"[ROBOT] Error activando normal mode: {e}")

    def smooth_axis(self, new_val, prev_val, alpha):
        return prev_val * alpha + new_val * (1.0 - alpha)

    async def send_move(self, x=0.0, y=0.0, z=0.0):
        x = self.smooth_axis(x, self.prev_x, SMOOTH_X)
        y = self.smooth_axis(y, self.prev_y, SMOOTH_Y)
        z = self.smooth_axis(z, self.prev_z, SMOOTH_Z)

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
                        "z": 0.0,
                    },
                },
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

    async def enable_lidar(self):
        if not USE_LIDAR:
            return

        try:
            print("[LIDAR] Activando...")
            await self.conn.datachannel.disableTrafficSaving(True)
            self.conn.datachannel.set_decoder(decoder_type="libvoxel")
            self.conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
            self.conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", lidar_callback)
            print("[LIDAR] Suscripto a voxel_map_compressed")
        except Exception as e:
            print(f"[LIDAR] No se pudo activar: {e}")

    async def control_loop(self):
        await self.ensure_normal_mode()
        await self.enable_lidar()
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
                    lidar_txt = "None"
                    if info.get("lidar_distance") is not None:
                        lidar_txt = f"{info['lidar_distance']:.2f}"

                    print(
                        f"[FOLLOW] x={info['x']:.2f} z={info['z']:.2f} "
                        f"err_x={info['err_x']:.2f} lidar={lidar_txt} "
                        f"mode={info['mode']}"
                    )
            else:
                lost_for = now - last_seen
                if lost_for > TARGET_LOST_TIMEOUT:
                    await self.hard_stop()
                    self.last_cmd_time = now


# ============================================================
# MAIN
# ============================================================
def main():
    face_memory = FaceMemory(similarity_threshold=0.55)
    frame_queue = queue.Queue(maxsize=10)

    # Elegí una conexión:
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.20.30.30")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
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
                    follow_info=follow_info,
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
                    print("[KEY] Reforzar modo normal")
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