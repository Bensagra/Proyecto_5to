"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     UNITREE GO2 - SISTEMA MEJORADO DE SEGUIMIENTO CON LIDAR + CÁMARA        ║
║     Person/Dog Follower: Detección SSD + Control PD + Evitación LiDAR      ║
║                                                                              ║
║     Características:                                                         ║
║     ✓ Detección de persona con SSD MobileNet                               ║
║     ✓ Control PD mejorado para seguimiento estable                         ║
║     ✓ Análisis de LiDAR 4D para evitar obstáculos                          ║
║     ✓ Threading optimizado (asyncio + worker threads)                      ║
║     ✓ Visualización en tiempo real con debug info                          ║
║     ✓ Manejo robusto de errores y reconexión                               ║
║                                                                              ║
║     USO:                                                                    ║
║     python seguimiento_perro_mejorado.py --ip 192.168.8.181                ║
║     python seguimiento_perro_mejorado.py --ip 192.168.8.181 --debug        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import argparse
import threading
import queue
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack


# ════════════════════════════════════════════════════════════════════════════
#  LOGGING Y CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DogFollower")


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Parámetros de configuración del sistema."""
    
    # ── Conexión WebRTC ──────────────────────────────────────────────────
    serial_number: str = "B42D2000P7I9GF8A"
    connection_method: str = "STA"  # "AP", "STA", "RemoteSTA"
    
    # ── Detección de persona (cámara) ────────────────────────────────────
    confidence_threshold: float = 0.55
    person_class_id: int = 1  # Clase persona en SSD
    frame_max_width: int = 1280
    frame_max_height: int = 720
    
    # ── Seguimiento (cámara) ─────────────────────────────────────────────
    desired_box_width_ratio: float = 0.24   # ancho deseado relativo a pantalla
    stop_box_width_ratio: float = 0.42      # umbral para detener (muy cerca)
    center_dead_zone: float = 0.035         # zona muerta en el centro
    
    # ── Control de velocidad ─────────────────────────────────────────────
    max_forward_speed: float = 2.0          # m/s adelante/atrás
    min_forward_speed: float = 1.0          # m/s velocidad mínima
    max_turn_speed: float = 0.6             # rad/s giro máximo
    
    # ── Ganancias PD para control ────────────────────────────────────────
    kp_forward: float = 10.0                # proporcional distancia
    kp_turn: float = 0.8                    # proporcional giro (antes 1.0)
    smoothing_alpha: float = 0.18           # suavizado exponencial (0-1)
    smooth_error_alpha: float = 0.75        # suavizado de error medido
    
    # ── LiDAR para evitar obstáculos ─────────────────────────────────────
    lidar_enabled: bool = True
    lidar_emergency_stop_m: float = 0.45    # obstáculo < 45cm → STOP total
    lidar_slow_zone_m: float = 0.90         # obstáculo < 90cm → reducir vel
    lidar_front_angle_deg: float = 60.0     # cono frontal analizado
    lidar_height_min: float = -0.15         # filtro altura mínima
    lidar_height_max: float = 1.80          # filtro altura máxima
    lidar_min_points: int = 10              # mínimo puntos para análisis válido
    
    # ── Timeouts ─────────────────────────────────────────────────────────
    person_lost_timeout_s: float = 0.5
    command_interval: float = 0.04          # Hz del loop de control (~25Hz)
    target_lost_search_enabled: bool = False
    target_lost_search_speed: float = 0.22
    
    # ── Mirror/Invert ────────────────────────────────────────────────────
    mirror_image: bool = False
    invert_turn: bool = True                # Si gira al revés, cambiar a True
    
    # ── Guardado de caras ────────────────────────────────────────────────
    save_face_crops: bool = True
    face_save_dir: Path = Path("capturas_caras")
    min_seconds_between_saves: float = 2.0
    face_similarity_threshold: float = 0.6
    
    # ── UI/Debug ─────────────────────────────────────────────────────────
    window_name: str = "Unitree Go2 - Dog Follower (Mejorado)"
    debug_prints: bool = True
    draw_lidar_data: bool = False


# ════════════════════════════════════════════════════════════════════════════
#  ESTADOS Y PRIMITIVOS
# ════════════════════════════════════════════════════════════════════════════

class RobotState(Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    FOLLOWING = "following"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class FollowCommand:
    """Comando de movimiento calculado."""
    x: float = 0.0           # adelante/atrás
    y: float = 0.0           # lateral
    z: float = 0.0           # giro
    mode: str = "idle"
    error_x: float = 0.0     # error horizontal
    box_width_ratio: float = 0.0
    sensor_state: str = "ok"


# ════════════════════════════════════════════════════════════════════════════
#  MEMORY PARA RECONOCIMIENTO DE CARAS
# ════════════════════════════════════════════════════════════════════════════

class FaceMemory:
    """Simple face embedding memory para evitar guardar la misma cara múltiples veces."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.embeddings = []
        self.threshold = similarity_threshold
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Genera embedding simple de una cara (64x64 greyscale normalizado)."""
        if face_img is None or face_img.size == 0:
            return np.zeros(4096)
        face = cv2.resize(face_img, (64, 64))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype(np.float32) / 255.0
        return face.flatten()
    
    def is_new_face(self, face_img: np.ndarray) -> bool:
        """Retorna True si es una cara no vista antes."""
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


# ════════════════════════════════════════════════════════════════════════════
#  DETECTOR DE PERSONA (SSD)
# ════════════════════════════════════════════════════════════════════════════

class PersonDetector:
    """Detección de persona usando SSD Lite MobileNet v3."""
    
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info(f"[DETECTOR] Usando device: {self.device}")
        
        # SSD Lite es más rápido que SSD completo
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()
        log.info("[DETECTOR] Modelo SSD cargado ✓")
    
    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Detecta personas en el frame.
        Retorna lista de dicts: [{"box": [x1,y1,x2,y2], "score": float}, ...]
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(frame_rgb).to(self.device)
        
        outputs = self.model([tensor])[0]
        
        boxes = outputs["boxes"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            # class_id 1 = person en COCO
            if label == 1 and score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.astype(int).tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score)
                })
        
        return detections


# ════════════════════════════════════════════════════════════════════════════
#  DETECTOR DE CARAS
# ════════════════════════════════════════════════════════════════════════════

class FaceCropper:
    """Extrae caras de dentro de bboxes de personas detectadas."""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_face_in_person(self, frame_bgr: np.ndarray, person_box: list) -> Tuple:
        """
        Detecta cara dentro de un bbox de persona.
        Retorna (face_crop, global_box) o (None, None) si no hay cara.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = person_box
        
        # Clamp a bordes
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        # ROI de la persona
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
        
        # Cara más grande
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        
        # Padding
        pad_x = int(fw * 0.15)
        pad_y = int(fh * 0.20)
        
        fx1 = max(0, fx - pad_x)
        fy1 = max(0, fy - pad_y)
        fx2 = min(person_roi.shape[1], fx + fw + pad_x)
        fy2 = min(person_roi.shape[0], fy + fh + pad_y)
        
        face_crop = person_roi[fy1:fy2, fx1:fx2]
        global_box = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
        
        return face_crop, global_box


# ════════════════════════════════════════════════════════════════════════════
#  ANALIZADOR DE LIDAR
# ════════════════════════════════════════════════════════════════════════════

class LidarAnalyzer:
    """Analiza el PointCloud del LiDAR 4D para detectar obstáculos."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def analyze(self, points: np.ndarray) -> dict:
        """
        Analiza punto cloud.
        Retorna dict:
          - min_front_dist: distancia mínima al obstáculo frontal (m)
          - obstacle_state: "clear" | "slow" | "emergency"
          - points_analyzed: cantidad de puntos procesados
        """
        if points is None or len(points) < self.cfg.lidar_min_points:
            return {
                "min_front_dist": 99.0,
                "obstacle_state": "clear",
                "points_analyzed": 0
            }
        
        # Filtro altura (ignorar suelo y techo)
        h_mask = (points[:, 2] > self.cfg.lidar_height_min) & \
                 (points[:, 2] < self.cfg.lidar_height_max)
        pts = points[h_mask]
        
        if len(pts) < self.cfg.lidar_min_points:
            return {
                "min_front_dist": 99.0,
                "obstacle_state": "clear",
                "points_analyzed": len(pts)
            }
        
        # x = adelante, y = izquierda en coords robot Go2
        angles_deg = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        
        half_angle = self.cfg.lidar_front_angle_deg / 2.0
        
        # Sector frontal
        front_mask = np.abs(angles_deg) < half_angle
        front_dists = dists[front_mask]
        min_front = float(np.min(front_dists)) if len(front_dists) > 0 else 99.0
        
        # Determinar estado del obstáculo
        if min_front < self.cfg.lidar_emergency_stop_m:
            obstacle_state = "emergency"
        elif min_front < self.cfg.lidar_slow_zone_m:
            obstacle_state = "slow"
        else:
            obstacle_state = "clear"
        
        return {
            "min_front_dist": min_front,
            "obstacle_state": obstacle_state,
            "points_analyzed": len(pts)
        }


# ════════════════════════════════════════════════════════════════════════════
#  CONTROLADOR DE MOVIMIENTO (RECONSTRUIDO)
# ════════════════════════════════════════════════════════════════════════════

class ImprovedFollowController:
    """
    Controlador mejorado de seguimiento con:
    - Control PD suavizado
    - Integración de LiDAR
    - Suavizado de comandos
    - Estados bien definidos
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Estado del controlador
        self.last_error_x = 0.0
        self.last_turn_sign = 0
        self.last_forward_speed = 0.0
        
        # Para suavizado
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0
    
    def compute_command(self, 
                       target_det: dict,
                       frame_shape: Tuple[int, int],
                       lidar_info: dict) -> FollowCommand:
        """
        Calcula comando de movimiento RECONSTRUIDO con buena lógica.
        
        Args:
            target_det: {"box": [x1,y1,x2,y2], "score": float}
            frame_shape: (height, width)
            lidar_info: {"min_front_dist": float, "obstacle_state": str, ...}
        
        Returns:
            FollowCommand con vx, vy, vz suavizados
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = target_det["box"]
        
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        
        # ─── CÁLCULO DEL ERROR HORIZONTAL ──────────────────────────────
        center_x = (x1 + x2) / 2.0
        err_x_raw = (center_x - (w / 2.0)) / (w / 2.0)
        
        if self.cfg.mirror_image:
            err_x_raw = -err_x_raw
        
        # Suavizar error medido
        err_x = (self.last_error_x * self.cfg.smooth_error_alpha + 
                 err_x_raw * (1.0 - self.cfg.smooth_error_alpha))
        self.last_error_x = err_x
        
        rel_box_w = box_w / float(w)
        rel_box_h = box_h / float(h)
        
        # ─── CÁLCULO DEL GIRO (MEJORADO) ───────────────────────────────
        z = 0.0
        abs_err = abs(err_x)
        
        if abs_err > self.cfg.center_dead_zone:
            # Error significativo
            norm_err = (abs_err - self.cfg.center_dead_zone) / \
                      (1.0 - self.cfg.center_dead_zone)
            norm_err = np.clip(norm_err, 0.0, 1.0)
            
            # Curva suave: comienza leve, sube más al final
            turn_strength = norm_err ** 1.7
            
            # Rango de giro
            z_mag = 0.10 + turn_strength * (self.cfg.max_turn_speed - 0.10)
            
            turn_sign = -1.0 if err_x < 0 else 1.0
            
            # Histéresis: no cambiar de dirección por micro-ruido
            if self.last_turn_sign != 0 and abs_err < 0.14:
                turn_sign = self.last_turn_sign
            
            self.last_turn_sign = turn_sign
            z = self.cfg.kp_turn * turn_sign * z_mag
        else:
            z = 0.0
            self.last_turn_sign = 0
        
        if self.cfg.invert_turn:
            z = -z
        
        # ─── CÁLCULO DE LA VELOCIDAD FORWARD (MEJORADO) ─────────────────
        distance_error = self.cfg.desired_box_width_ratio - rel_box_w
        too_close = (rel_box_w >= self.cfg.stop_box_width_ratio or 
                     rel_box_h >= 0.78)
        
        mode = "idle"
        x = 0.0
        
        if too_close:
            # Muy cerca: detener
            x = 0.0
            mode = "TOO_CLOSE"
        else:
            # Velocidad proporcional a cuán lejos está
            raw_x = max(0.0, distance_error * self.cfg.kp_forward)
            
            # Penalización por giro (no avanzar mucho mientras gira)
            if abs_err < 0.10:
                turn_penalty = 1.0
            elif abs_err < 0.22:
                turn_penalty = 0.88
            elif abs_err < 0.35:
                turn_penalty = 0.72
            else:
                turn_penalty = 0.50
            
            raw_x *= turn_penalty
            
            # Velocidad según distancia
            if distance_error > 0.12:
                x = np.clip(raw_x, 1.2, self.cfg.max_forward_speed)
                mode = "RUN"
            elif distance_error > 0.08:
                x = np.clip(raw_x, 0.50, self.cfg.max_forward_speed * 0.85)
                mode = "FAST_FOLLOW"
            elif distance_error > 0.025:
                x = np.clip(raw_x, self.cfg.min_forward_speed, 0.65)
                mode = "SLOW_FOLLOW"
            elif distance_error > -0.015:
                x = 0.0
                mode = "HOLD"
            else:
                x = -0.2  # retroceder levemente si muy cerca
                mode = "BACK_OFF"
        
        # ─── INTEGRACIÓN DE LIDAR (EVITAR OBSTÁCULOS) ──────────────────
        sensor_state = "OK"
        lidar_state = lidar_info.get("obstacle_state", "clear")
        
        if lidar_state == "emergency":
            # PARAR TODO
            x = 0.0
            z = 0.0
            mode = f"{mode}+LIDAR_EMERGENCY"
            sensor_state = "EMERGENCY"
        elif lidar_state == "slow":
            # Reducir velocidad, mantener giro
            x *= 0.5
            mode = f"{mode}+LIDAR_SLOW"
            sensor_state = "SLOW"
        
        # ─── SUAVIZADO FINAL DE COMANDOS ───────────────────────────────
        x_smooth = (self.prev_x * self.cfg.smoothing_alpha + 
                   x * (1.0 - self.cfg.smoothing_alpha))
        y_smooth = (self.prev_y * self.cfg.smoothing_alpha + 
                   0.0 * (1.0 - self.cfg.smoothing_alpha))
        z_smooth = (self.prev_z * self.cfg.smoothing_alpha + 
                   z * (1.0 - self.cfg.smoothing_alpha))
        
        self.prev_x = x_smooth
        self.prev_y = y_smooth
        self.prev_z = z_smooth
        
        return FollowCommand(
            x=float(x_smooth),
            y=float(y_smooth),
            z=float(z_smooth),
            mode=mode,
            error_x=float(err_x),
            box_width_ratio=float(rel_box_w),
            sensor_state=sensor_state
        )
    
    def hard_stop(self) -> FollowCommand:
        """Detiene todos los movimientos."""
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0
        self.last_error_x = 0.0
        self.last_turn_sign = 0
        
        return FollowCommand(x=0.0, y=0.0, z=0.0, mode="HARD_STOP", 
                            sensor_state="STOPPED")


# ════════════════════════════════════════════════════════════════════════════
#  COMUNICACIÓN CON ROBOT
# ════════════════════════════════════════════════════════════════════════════

class RobotCommandSender:
    """Envía comandos al robot de forma segura."""
    
    def __init__(self, conn: Go2WebRTCConnection):
        self.conn = conn
        self.motion_ready = False
        self.last_cmd_time = 0.0
    
    async def ensure_normal_mode(self):
        """Verifica que el robot esté en modo normal."""
        if self.motion_ready:
            return
        
        try:
            log.info("[ROBOT] Verificando modo de movimiento...")
            response = await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1001}
            )
            
            code = response["data"]["header"]["status"]["code"]
            
            if code == 0:
                data = json.loads(response["data"]["data"])
                current_mode = data.get("name", "unknown")
                log.info(f"[ROBOT] Modo actual: {current_mode}")
                
                if current_mode != "normal":
                    log.info("[ROBOT] Cambiando a modo normal...")
                    await self.conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"],
                        {
                            "api_id": 1002,
                            "parameter": {"name": "normal"}
                        }
                    )
                    await asyncio.sleep(4.0)
                    log.info("[ROBOT] Modo normal activado ✓")
            else:
                log.warning("[ROBOT] No pude leer motion mode, continuando...")
            
            self.motion_ready = True
        
        except Exception as e:
            log.error(f"[ROBOT] Error en ensure_normal_mode: {e}")
    
    async def send_move_command(self, cmd: FollowCommand, 
                               min_interval: float = 0.04) -> bool:
        """
        Envía comando de movimiento al robot.
        Respeta intervalo mínimo entre comandos.
        """
        now = time.time()
        if now - self.last_cmd_time < min_interval:
            return False
        
        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": float(cmd.x),
                        "y": float(cmd.y),
                        "z": float(cmd.z)
                    }
                }
            )
            self.last_cmd_time = now
            return True
        
        except Exception as e:
            log.error(f"[ROBOT] Error enviando move: {e}")
            return False
    
    async def stop(self):
        """Detiene el robot."""
        cmd = FollowCommand(x=0.0, y=0.0, z=0.0, mode="STOP")
        await self.send_move_command(cmd)


# ════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════════════════

def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp a rango [lo, hi]."""
    return max(lo, min(hi, val))


def choose_best_target(detections: list, frame_shape: Tuple[int, int]) -> Tuple:
    """
    Elige la mejor persona detectada según:
    - Confianza
    - Tamaño (más grande = más cercano)
    - Posición (preferir centrada)
    """
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
        
        # Scoring: confianza + tamaño relativo - distancia del centro
        value = score * 2.0 + (area / (w * h)) * 2.5 - center_dist * 1.2
        
        if value > best_value:
            best_value = value
            best_idx = i
    
    if best_idx is not None:
        return best_idx, detections[best_idx]
    return None, None


def save_face_crop(face_crop: np.ndarray, save_dir: Path) -> Optional[Path]:
    """Guarda una cara cortada."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    filename = save_dir / f"cara_{timestamp}_{millis:03d}.jpg"
    
    cv2.imwrite(str(filename), face_crop)
    return filename


def draw_debug_info(frame: np.ndarray,
                   detections: list,
                   face_boxes: list,
                   cmd: FollowCommand,
                   lidar_info: dict,
                   fps: float,
                   target_idx: Optional[int] = None) -> np.ndarray:
    """Dibuja información de debug en el frame."""
    output = frame.copy()
    h, w = output.shape[:2]
    
    # ─── Línea central ─────────────────────────────────────────────────
    cx = w // 2
    cv2.line(output, (cx, 0), (cx, h), (255, 255, 0), 1)
    
    # ─── Detecciones ───────────────────────────────────────────────────
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        
        color = (0, 255, 0)
        if i == target_idx:
            color = (0, 165, 255)
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"P:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, max(0, y1 - th - 8)), 
                     (x1 + tw + 6, y1), color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    # ─── Caras detectadas ──────────────────────────────────────────────
    if face_boxes:
        for (fx1, fy1, fx2, fy2) in face_boxes:
            cv2.rectangle(output, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(output, "Face", (fx1, max(20, fy1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    
    # ─── FPS ───────────────────────────────────────────────────────────
    cv2.putText(output, f"FPS: {fps:.1f}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    
    # ─── Conteo ────────────────────────────────────────────────────────
    cv2.putText(output, f"People: {len(detections)}", (20, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    
    # ─── Comando actual ────────────────────────────────────────────────
    if cmd:
        text = (f"CMD: x={cmd.x:+.2f} z={cmd.z:+.2f} "
               f"box={cmd.box_width_ratio:.2f} mode={cmd.mode}")
        cv2.putText(output, text, (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
    
    # ─── LiDAR info ────────────────────────────────────────────────────
    if lidar_info:
        lidar_text = (f"LIDAR: {lidar_info.get('min_front_dist', 0):.2f}m "
                     f"[{lidar_info.get('obstacle_state', 'N/A')}]")
        color_lidar = (0, 255, 0)
        if lidar_info.get('obstacle_state') == 'emergency':
            color_lidar = (0, 0, 255)
        elif lidar_info.get('obstacle_state') == 'slow':
            color_lidar = (0, 165, 255)
        
        cv2.putText(output, lidar_text, (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_lidar, 2, cv2.LINE_AA)
    
    return output


# ════════════════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL (ASYNCIO)
# ════════════════════════════════════════════════════════════════════════════

async def main_async(cfg: Config, conn: Go2WebRTCConnection):
    """Loop principal en asyncio."""
    
    # Inicializar componentes
    log.info("[MAIN] Inicializando componentes...")
    
    detector = PersonDetector(cfg.confidence_threshold)
    face_cropper = FaceCropper()
    face_memory = FaceMemory(cfg.face_similarity_threshold)
    lidar_analyzer = LidarAnalyzer(cfg)
    controller = ImprovedFollowController(cfg)
    robot_sender = RobotCommandSender(conn)
    
    # Estado compartido
    frame_queue = queue.Queue(maxsize=10)
    lidar_queue = queue.Queue(maxsize=5)
    
    # Conectar a WebRTC
    log.info("[MAIN] Conectando a WebRTC...")
    await conn.connect()
    log.info("[MAIN] Conectado ✓")
    
    # Activar cámara
    conn.video.switchVideoChannel(True)
    
    # Activar LiDAR si está habilitado
    if cfg.lidar_enabled:
        try:
            await conn.datachannel.disableTrafficSaving(True)
            conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
            log.info("[LIDAR] Sensor activado")
        except Exception as e:
            log.warning(f"[LIDAR] No se pudo activar: {e}")
    
    # Preparar robot
    await robot_sender.ensure_normal_mode()
    
    # ─── Callback de cámara ────────────────────────────────────────────
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                frame_queue.put(img)
            
            except Exception as e:
                log.debug(f"[CAMERA] Error en recv: {e}")
                break
    
    # ─── Callback de LiDAR ─────────────────────────────────────────────
    def lidar_callback(message: dict):
        try:
            if not cfg.lidar_enabled:
                return
            
            data = message.get("data", {})
            positions = data.get("data", {}).get("positions", [])
            
            # Convertir a numpy array
            if isinstance(positions, list) and len(positions) > 0:
                # Agrupar en [x,y,z] tuples
                points = np.array([
                    positions[i:i+3] 
                    for i in range(0, len(positions), 3)
                    if i+2 < len(positions)
                ], dtype=np.float32)
                
                if len(points) > 0:
                    if lidar_queue.full():
                        try:
                            lidar_queue.get_nowait()
                        except queue.Empty:
                            pass
                    lidar_queue.put(points)
        
        except Exception as e:
            log.debug(f"[LIDAR] Error en callback: {e}")
    
    # Suscribirse a streams
    conn.video.add_track_callback(recv_camera_stream)
    
    if cfg.lidar_enabled:
        try:
            conn.datachannel.pub_sub.subscribe(
                "rt/utlidar/voxel_map_compressed",
                lidar_callback
            )
        except Exception as e:
            log.warning(f"[LIDAR] No se pudo suscribir: {e}")
    
    log.info("[MAIN] Streams configurados ✓")
    
    # ─── Loop de control principal ──────────────────────────────────────
    prev_time = time.time()
    last_save_time = 0.0
    last_cmd_time = 0.0
    
    try:
        while True:
            now = time.time()
            
            # ─ Obtener frame ─
            frame = None
            try:
                frame = frame_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            
            # ─ Obtener LiDAR ─
            lidar_points = None
            try:
                lidar_points = lidar_queue.get_nowait()
            except queue.Empty:
                pass
            
            # ─ Detección ─
            detections = detector.detect(frame)
            face_boxes = []
            target_idx = None
            target_det = None
            cmd = controller.hard_stop()
            lidar_info = {"min_front_dist": 99.0, "obstacle_state": "clear"}
            
            # ─ Análisis LiDAR ─
            if lidar_points is not None:
                lidar_info = lidar_analyzer.analyze(lidar_points)
            
            # ─ Elegir target ─
            if len(detections) > 0:
                target_idx, target_det = choose_best_target(detections, frame.shape)
                
                # ─ Calcular comando ─
                if target_det is not None:
                    cmd = controller.compute_command(
                        target_det,
                        frame.shape,
                        lidar_info
                    )
                    
                    # ─ Enviar comando al robot ─
                    if now - last_cmd_time >= cfg.command_interval:
                        await robot_sender.send_move_command(cmd, cfg.command_interval)
                        last_cmd_time = now
                        
                        if cfg.debug_prints:
                            log.info(
                                f"[CMD] x={cmd.x:+.2f} z={cmd.z:+.2f} "
                                f"err={cmd.error_x:+.2f} box={cmd.box_width_ratio:.2f} "
                                f"mode={cmd.mode}"
                            )
                    
                    # ─ Procesar caras ─
                    if cfg.save_face_crops:
                        person_box = target_det["box"]
                        face_crop, face_box = face_cropper.detect_face_in_person(
                            frame, person_box
                        )
                        
                        if face_box is not None:
                            face_boxes.append(face_box)
                        
                        if (face_crop is not None and face_crop.size > 0 and
                            (now - last_save_time) >= cfg.min_seconds_between_saves):
                            is_new = face_memory.is_new_face(face_crop)
                            if is_new:
                                path = save_face_crop(face_crop, cfg.face_save_dir)
                                log.info(f"[FACE] Nueva cara guardada: {path}")
                                last_save_time = now
            
            # ─ FPS y visualización ─
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            
            output = draw_debug_info(
                frame,
                detections,
                face_boxes,
                cmd,
                lidar_info,
                fps,
                target_idx
            )
            
            cv2.imshow(cfg.window_name, output)
            
            # ─ Teclas ─
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info("[UI] Quit solicitado")
                break
            elif key == ord('s'):
                log.info("[UI] Parada manual")
                await robot_sender.stop()
            elif key == ord('n'):
                log.info("[UI] Reestableciendo modo normal")
                robot_sender.motion_ready = False
                await robot_sender.ensure_normal_mode()
    
    except KeyboardInterrupt:
        log.info("[MAIN] Interrumpido por usuario")
    
    finally:
        log.info("[MAIN] Limpiando...")
        await robot_sender.stop()
        cv2.destroyAllWindows()
        try:
            await conn.disconnect()
        except:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point principal."""
    
    parser = argparse.ArgumentParser(
        description="Unitree Go2 - Dog Follower (Mejorado con LiDAR)"
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.8.181",
        help="IP del robot (default: 192.168.8.181)"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="B42D2000P7I9GF8A",
        help="Serial number del robot (default: B42D2000P7I9GF8A)"
    )
    parser.add_argument(
        "--ap",
        action="store_true",
        help="Usar LocalAP en lugar de LocalSTA"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug verbose"
    )
    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Desactivar LiDAR"
    )
    parser.add_argument(
        "--no-face-save",
        action="store_true",
        help="No guardar caras"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear config
    cfg = Config()
    cfg.robot_ip = args.ip
    cfg.serial_number = args.serial
    cfg.connection_method = "AP" if args.ap else "STA"
    cfg.lidar_enabled = not args.no_lidar
    cfg.save_face_crops = not args.no_face_save
    cfg.debug_prints = args.debug
    
    log.info("╔════════════════════════════════════════════════════════════════╗")
    log.info("║     UNITREE GO2 - DOG FOLLOWER (MEJORADO CON LIDAR)           ║")
    log.info("╚════════════════════════════════════════════════════════════════╝")
    log.info(f"[CONFIG] Serial: {cfg.serial_number}")
    log.info(f"[CONFIG] Método: {cfg.connection_method}")
    log.info(f"[CONFIG] LiDAR: {'✓ Habilitado' if cfg.lidar_enabled else '✗ Deshabilitado'}")
    log.info(f"[CONFIG] Guardar caras: {'✓' if cfg.save_face_crops else '✗'}")
    
    # Crear conexión
    try:
        if cfg.connection_method == "AP":
            conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        else:
            # Siempre usar Serial Number (STA)
            conn = Go2WebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                serialNumber=cfg.serial_number
            )
    except Exception as e:
        log.error(f"Error creando conexión: {e}")
        return
    
    # Ejecutar
    try:
        asyncio.run(main_async(cfg, conn))
    except Exception as e:
        log.error(f"Error en main_async: {e}", exc_info=True)
    finally:
        log.info("[MAIN] Terminado.")


if __name__ == "__main__":
    main()
