"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         UNITREE GO2 - SISTEMA AUTÓNOMO DE SEGUIMIENTO PARA RESCATES         ║
║         Person Follower using Camera (SSD) + LiDAR Obstacle Avoidance       ║
║         Librería: unitree_webrtc_connect (legion1581)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALACIÓN RÁPIDA:
  pip install unitree_webrtc_connect opencv-python-headless numpy

Para descargar el modelo SSD:
  wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel
  wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt

USO:
  python rescue_dog_follower.py --ip 192.168.8.181
  python rescue_dog_follower.py --ip 192.168.8.181 --target-distance 1.5 --max-speed 0.6

ARQUITECTURA:
  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────┐
  │ Cámara RGB  │───▶│  SSD Detector    │───▶│                            │
  └─────────────┘    │  (MobileNet-SSD) │    │   PersonFollowController   │
                     └──────────────────┘    │   (PD Control Loop)        │
  ┌─────────────┐    ┌──────────────────┐    │                            │
  │ LiDAR 4D    │───▶│ Obstacle Analyzer│───▶│   ──── vx, vy, vyaw ────  │
  └─────────────┘    └──────────────────┘    └──────────┬─────────────────┘
                                                         │
                                             ┌───────────▼──────────────┐
                                             │  UnitreeWebRTCConnection │
                                             │  sport/request API       │
                                             └──────────────────────────┘
"""

import asyncio
import json
import time
import logging
import argparse
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

import cv2
import numpy as np

# ── Unitree WebRTC imports ──────────────────────────────────────────────────
# go2_webrtc_connect (fork local) usa Go2WebRTCConnection
try:
    from unitree_webrtc_connect import UnitreeWebRTCConnection as Go2WebRTCConnection, WebRTCConnectionMethod
except ImportError:
    from unitree_webrtc_connect import Go2WebRTCConnection, WebRTCConnectionMethod

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RescueDog")

# ════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Conexión ──────────────────────────────────────────────────────────
    robot_ip: str = "192.168.8.181"
    connection_method: str = "STA"          # "AP", "STA", "Remote"

    # ── Detección de persona (cámara) ─────────────────────────────────────
    ssd_prototxt:   str = "MobileNetSSD_deploy.prototxt"
    ssd_model:      str = "MobileNetSSD_deploy.caffemodel"
    ssd_confidence: float = 0.55           # umbral mínimo de confianza
    frame_width:    int = 640
    frame_height:   int = 480

    # ── Seguimiento ───────────────────────────────────────────────────────
    target_distance: float = 1.2           # metros: distancia ideal a la persona
    follow_zone_px:  int = 80              # píxeles: tolerancia centro horizontal

    # ── Límites de velocidad ──────────────────────────────────────────────
    max_vx:    float = 0.5                 # m/s adelante/atrás
    max_vy:    float = 0.3                 # m/s lateral (strafe)
    max_vyaw:  float = 0.6                 # rad/s giro

    # ── Ganancias del controlador PD ──────────────────────────────────────
    kp_yaw:  float = 0.0025               # proporcional para giro
    kd_yaw:  float = 0.0005               # derivativo para giro
    kp_dist: float = 0.6                  # proporcional para avance
    kd_dist: float = 0.1                  # derivativo para avance

    # ── Seguridad LiDAR ───────────────────────────────────────────────────
    lidar_emergency_stop_m:  float = 0.45  # obstáculo < 45 cm → STOP total
    lidar_slow_zone_m:       float = 0.90  # obstáculo < 90 cm → reducir vel
    lidar_front_angle_deg:   float = 60.0  # cono frontal analizado (±30°)
    lidar_height_min:        float = -0.15 # filtro altura mínima (m)
    lidar_height_max:        float = 1.80  # filtro altura máxima (m)

    # ── Timeouts ──────────────────────────────────────────────────────────
    person_lost_timeout_s: float = 3.0    # segundos sin detección → stand still
    control_hz: float = 15.0              # frecuencia del loop de control


# ════════════════════════════════════════════════════════════════════════════
#  ESTADO COMPARTIDO (thread-safe)
# ════════════════════════════════════════════════════════════════════════════

class RobotState(Enum):
    IDLE       = "idle"
    SEARCHING  = "searching"
    FOLLOWING  = "following"
    STOPPING   = "stopping"
    EMERGENCY  = "emergency"


@dataclass
class SharedState:
    """Datos intercambiados entre callbacks de sensores y el controlador."""
    # Cámara
    person_bbox:        Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h)
    frame_width:        int = 640
    person_detected_at: float = 0.0
    person_confidence:  float = 0.0

    # LiDAR
    min_front_dist:     float = 99.0      # distancia mínima sector frontal (m)
    lidar_points:       Optional[np.ndarray] = None

    # Control
    robot_state:        RobotState = RobotState.IDLE
    last_vx:            float = 0.0
    last_vyaw:          float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_person(self, bbox, confidence, frame_w):
        with self._lock:
            self.person_bbox        = bbox
            self.person_confidence  = confidence
            self.frame_width        = frame_w
            self.person_detected_at = time.time()

    def clear_person(self):
        with self._lock:
            self.person_bbox = None

    def update_lidar(self, min_dist: float, points: np.ndarray):
        with self._lock:
            self.min_front_dist = min_dist
            self.lidar_points   = points

    def snapshot(self):
        with self._lock:
            return (
                self.person_bbox,
                self.frame_width,
                self.person_detected_at,
                self.min_front_dist,
                self.robot_state,
                self.last_vx,
                self.last_vyaw,
            )


# ════════════════════════════════════════════════════════════════════════════
#  DETECTOR SSD (corre en hilo separado para no bloquear asyncio)
# ════════════════════════════════════════════════════════════════════════════

class PersonDetector:
    """
    MobileNet-SSD ejecutado en un ThreadPoolExecutor.
    Clase 15 = person en el modelo COCO/VOC del SSD preentrenado.
    """
    PERSON_CLASS = 15

    def __init__(self, cfg: Config):
        self.cfg = cfg
        log.info("Cargando modelo SSD desde %s / %s", cfg.ssd_prototxt, cfg.ssd_model)
        self.net = cv2.dnn.readNetFromCaffe(cfg.ssd_prototxt, cfg.ssd_model)
        # Usar GPU si está disponible (CUDA backend)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        log.info("Modelo SSD listo ✓")

    def detect_person(self, frame: np.ndarray):
        """
        Devuelve (bbox, confidence) de la persona más grande detectada,
        o (None, 0) si no hay detección.
        bbox = (x, y, w, h) en píxeles absolutos.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()  # shape: (1,1,N,7)

        best_bbox  = None
        best_conf  = 0.0
        best_area  = 0

        for i in range(detections.shape[2]):
            class_id   = int(detections[0, 0, i, 1])
            confidence = float(detections[0, 0, i, 2])

            if class_id != self.PERSON_CLASS:
                continue
            if confidence < self.cfg.ssd_confidence:
                continue

            # Coordenadas normalizadas → píxeles
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh  = x2 - x1, y2 - y1
            area    = bw * bh

            # Quedarse con la persona más grande (más cercana)
            if area > best_area and confidence > best_conf:
                best_area = area
                best_conf = confidence
                best_bbox = (x1, y1, bw, bh)

        return best_bbox, best_conf


# ════════════════════════════════════════════════════════════════════════════
#  ANALIZADOR DE LIDAR
# ════════════════════════════════════════════════════════════════════════════

class LidarAnalyzer:
    """
    Analiza el PointCloud del LiDAR 4D del Go2 para detectar obstáculos.
    El PointCloud viene como numpy array (N, 4) = [x, y, z, intensity].
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, points: np.ndarray) -> dict:
        """
        Devuelve dict con:
          - min_front_dist: distancia mínima al obstáculo en sector frontal (m)
          - obstacle_left: True si hay obstáculo cerca a la izquierda
          - obstacle_right: True si hay obstáculo cerca a la derecha
          - person_lidar_dist: estimación de distancia a persona frente (m)
        """
        if points is None or len(points) == 0:
            return {"min_front_dist": 99.0, "obstacle_left": False,
                    "obstacle_right": False}

        # Filtrar por altura (ignorar suelo y techo)
        h_mask = (points[:, 2] > self.cfg.lidar_height_min) & \
                 (points[:, 2] < self.cfg.lidar_height_max)
        pts = points[h_mask]

        if len(pts) == 0:
            return {"min_front_dist": 99.0, "obstacle_left": False,
                    "obstacle_right": False}

        # x = adelante, y = izquierda en coordenadas robot Go2
        # Calcular ángulo en plano horizontal
        angles_deg = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        dists      = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

        half_angle = self.cfg.lidar_front_angle_deg / 2.0

        # ── Sector FRONTAL ───────────────────────────────────────────────
        front_mask = np.abs(angles_deg) < half_angle
        front_dists = dists[front_mask]
        min_front   = float(np.min(front_dists)) if len(front_dists) > 0 else 99.0

        # ── Sectores LATERALES (para esquivar) ───────────────────────────
        slow_thresh = self.cfg.lidar_slow_zone_m
        left_mask   = (angles_deg > half_angle) & (angles_deg < half_angle + 45)
        right_mask  = (angles_deg < -half_angle) & (angles_deg > -(half_angle + 45))

        obs_left  = bool(len(dists[left_mask]) > 0  and np.min(dists[left_mask])  < slow_thresh)
        obs_right = bool(len(dists[right_mask]) > 0 and np.min(dists[right_mask]) < slow_thresh)

        return {
            "min_front_dist": min_front,
            "obstacle_left":  obs_left,
            "obstacle_right": obs_right,
        }


# ════════════════════════════════════════════════════════════════════════════
#  CONTROLADOR PD DE SEGUIMIENTO
# ════════════════════════════════════════════════════════════════════════════

class FollowController:
    """
    Controlador PD que calcula (vx, vy, vyaw) para seguir a la persona.
    
    Lógica:
      - Error de yaw  → error_px   = centro_persona_x - centro_imagen_x
      - Error de dist → error_dist = min_front_dist - target_distance
      - vyaw = Kp_yaw  * error_px   + Kd_yaw  * d(error_px)/dt
      - vx   = Kp_dist * error_dist + Kd_dist * d(error_dist)/dt
    """

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.prev_err_yaw  = 0.0
        self.prev_err_dist = 0.0
        self.prev_time     = time.time()

    def compute(
        self,
        person_bbox: Optional[Tuple],
        frame_width: int,
        min_front_dist: float,
        lidar_info: dict,
    ) -> Tuple[float, float, float]:
        """Devuelve (vx, vy, vyaw) clampeados a los límites configurados."""

        now = time.time()
        dt  = max(now - self.prev_time, 1e-3)
        self.prev_time = now

        cfg = self.cfg

        # ── Sin persona → parar ──────────────────────────────────────────
        if person_bbox is None:
            self.prev_err_yaw  = 0.0
            self.prev_err_dist = 0.0
            return 0.0, 0.0, 0.0

        # ── PARADA DE EMERGENCIA LiDAR ───────────────────────────────────
        if min_front_dist < cfg.lidar_emergency_stop_m:
            log.warning("⚠ EMERGENCY STOP: obstáculo a %.2f m", min_front_dist)
            self.prev_err_yaw  = 0.0
            self.prev_err_dist = 0.0
            return 0.0, 0.0, 0.0

        bx, by, bw, bh = person_bbox
        cx_person = bx + bw / 2.0
        cx_frame  = frame_width / 2.0

        # ── Error de YAW (izquierda/derecha) ─────────────────────────────
        err_yaw  = cx_frame - cx_person   # positivo = persona a izquierda → girar izq
        d_err_yaw = (err_yaw - self.prev_err_yaw) / dt
        vyaw = cfg.kp_yaw * err_yaw + cfg.kd_yaw * d_err_yaw
        self.prev_err_yaw = err_yaw

        # Si la persona está centrada (dentro de zona de tolerancia) no girar
        if abs(err_yaw) < cfg.follow_zone_px:
            vyaw = 0.0

        # ── Error de DISTANCIA (adelante/atrás) ──────────────────────────
        err_dist  = min_front_dist - cfg.target_distance
        d_err_dist = (err_dist - self.prev_err_dist) / dt
        vx = cfg.kp_dist * err_dist + cfg.kd_dist * d_err_dist
        self.prev_err_dist = err_dist

        # Si ya estamos en la zona objetivo, no avanzar
        if abs(err_dist) < 0.10:
            vx = 0.0

        # ── Factor de reducción por obstáculo cercano ─────────────────────
        if min_front_dist < cfg.lidar_slow_zone_m:
            slow_factor = (min_front_dist - cfg.lidar_emergency_stop_m) / \
                          (cfg.lidar_slow_zone_m - cfg.lidar_emergency_stop_m)
            slow_factor = max(0.0, min(1.0, slow_factor))
            vx   *= slow_factor
            vyaw *= slow_factor

        # ── Evasión lateral de obstáculos ─────────────────────────────────
        vy = 0.0
        if lidar_info.get("obstacle_left") and not lidar_info.get("obstacle_right"):
            vy = -cfg.max_vy * 0.4   # esquivar hacia derecha
        elif lidar_info.get("obstacle_right") and not lidar_info.get("obstacle_left"):
            vy = cfg.max_vy * 0.4    # esquivar hacia izquierda

        # ── Clamp velocidades ─────────────────────────────────────────────
        vx   = float(np.clip(vx,   -cfg.max_vx,   cfg.max_vx))
        vy   = float(np.clip(vy,   -cfg.max_vy,   cfg.max_vy))
        vyaw = float(np.clip(vyaw, -cfg.max_vyaw,  cfg.max_vyaw))

        return vx, vy, vyaw


# ════════════════════════════════════════════════════════════════════════════
#  ROBOT COMMANDER (envía comandos WebRTC)
# ════════════════════════════════════════════════════════════════════════════

class RobotCommander:
    """
    Envía comandos de movimiento al Go2 mediante la API sport/request
    a través del datachannel WebRTC.

    API IDs del Go2 Sport service:
      1001 = Damp (relax)
      1003 = StandUp
      1004 = StandDown  
      1006 = RecoveryStand
      1007 = StopMove
      1008 = Move (x, y, z=vyaw)
    """

    TOPIC_SPORT = "rt/api/sport/request"
    API_MOVE    = 1008
    API_STOP    = 1007
    API_STAND   = 1003

    def __init__(self, conn: UnitreeWebRTCConnection):
        self.conn    = conn
        self._msg_id = 1

    def _next_id(self):
        self._msg_id += 1
        return self._msg_id

    async def _send(self, api_id: int, params: dict):
        msg = {
            "type": "msg",
            "topic": self.TOPIC_SPORT,
            "data": json.dumps({
                "header": {
                    "identity": {
                        "id": self._next_id(),
                        "api_id": api_id
                    }
                },
                "parameter": json.dumps(params)
            })
        }
        await self.conn.datachannel.send(json.dumps(msg))

    async def move(self, vx: float, vy: float, vyaw: float):
        """Envía comando de movimiento continuo."""
        await self._send(self.API_MOVE, {"x": vx, "y": vy, "z": vyaw})

    async def stop(self):
        """Para el robot inmediatamente."""
        await self._send(self.API_STOP, {})

    async def stand_up(self):
        await self._send(self.API_STAND, {})


# ════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class RescueDogApp:
    """
    Orquesta todos los componentes:
      1. Conexión WebRTC al Go2
      2. Callbacks de cámara (→ SSD) y LiDAR (→ analyzer)
      3. Loop de control a cfg.control_hz Hz
      4. Visualización opcional por OpenCV
    """

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.state    = SharedState()
        self.detector = PersonDetector(cfg)
        self.analyzer = LidarAnalyzer(cfg)
        self.control  = FollowController(cfg)

        # Cola de frames para procesar SSD fuera del hilo asyncio
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)

        # Conexión WebRTC
        method_map = {
            "AP":     WebRTCConnectionMethod.LocalAP,
            "STA":    WebRTCConnectionMethod.LocalSTA,
            "Remote": WebRTCConnectionMethod.Remote,
        }
        method = method_map.get(cfg.connection_method, WebRTCConnectionMethod.LocalSTA)
        self.conn = Go2WebRTCConnection(method, ip=cfg.robot_ip)
        self.cmd  = RobotCommander(self.conn)

        self._running = False
        log.info("RescueDogApp inicializada — robot: %s", cfg.robot_ip)

    # ── Callbacks de sensores ────────────────────────────────────────────

    async def _on_video_frame(self, frame):
        """Recibe frame de la cámara (av.VideoFrame o numpy BGR)."""
        try:
            # Convertir si es av.VideoFrame
            if hasattr(frame, "to_ndarray"):
                img = frame.to_ndarray(format="bgr24")
            else:
                img = frame

            # Encolar para el hilo detector (no bloquear asyncio)
            if not self._frame_q.full():
                self._frame_q.put_nowait(img.copy())
        except Exception as e:
            log.debug("Error en callback de video: %s", e)

    def _on_lidar_data(self, point_cloud):
        """Recibe PointCloud del LiDAR (numpy array N×4 o N×3)."""
        try:
            pts = np.asarray(point_cloud, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                return

            lidar_info = self.analyzer.analyze(pts)
            self.state.update_lidar(
                min_dist=lidar_info["min_front_dist"],
                points=pts,
            )
        except Exception as e:
            log.debug("Error procesando LiDAR: %s", e)

    # ── Hilo detector SSD ────────────────────────────────────────────────

    def _detection_worker(self):
        """
        Hilo de Python puro (no asyncio) que toma frames de la cola
        y ejecuta MobileNet-SSD.
        """
        log.info("Hilo de detección SSD iniciado")
        while self._running:
            try:
                img = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            bbox, conf = self.detector.detect_person(img)

            if bbox is not None:
                h, w = img.shape[:2]
                self.state.update_person(bbox, conf, w)
                log.debug("Persona detectada conf=%.2f bbox=%s", conf, bbox)
            else:
                # Si hace mucho tiempo sin ver a nadie, limpiar
                if time.time() - self.state.person_detected_at > self.cfg.person_lost_timeout_s:
                    self.state.clear_person()

    # ── Loop de control principal ────────────────────────────────────────

    async def _control_loop(self):
        """
        Se ejecuta a cfg.control_hz Hz.
        Lee estado compartido y envía comandos de movimiento.
        """
        interval = 1.0 / self.cfg.control_hz
        log.info("Control loop iniciado a %.1f Hz", self.cfg.control_hz)

        # Dar tiempo al robot para levantarse
        await asyncio.sleep(2.0)
        await self.cmd.stand_up()
        await asyncio.sleep(1.5)
        log.info("Robot de pie — iniciando seguimiento")

        while self._running:
            t0 = time.time()

            (person_bbox,
             frame_width,
             detected_at,
             min_front_dist,
             robot_state,
             last_vx,
             last_vyaw) = self.state.snapshot()

            # ── LiDAR info completo ──────────────────────────────────────
            pts = self.state.lidar_points
            lidar_info = self.analyzer.analyze(pts) if pts is not None else {}

            # ── Emergencia LiDAR ─────────────────────────────────────────
            if min_front_dist < self.cfg.lidar_emergency_stop_m:
                with self.state._lock:
                    self.state.robot_state = RobotState.EMERGENCY
                await self.cmd.stop()
                log.warning("🚨 EMERGENCIA: obstáculo a %.2f m — STOP!", min_front_dist)
                await asyncio.sleep(interval)
                continue

            # ── Sin persona → mantener quieto ────────────────────────────
            if person_bbox is None:
                with self.state._lock:
                    self.state.robot_state = RobotState.SEARCHING
                # Girar suavemente buscando a la persona (rescate)
                await self.cmd.move(0.0, 0.0, 0.15)
                await asyncio.sleep(interval)
                continue

            # ── Calcular comandos ─────────────────────────────────────────
            vx, vy, vyaw = self.control.compute(
                person_bbox=person_bbox,
                frame_width=frame_width,
                min_front_dist=min_front_dist,
                lidar_info=lidar_info,
            )

            with self.state._lock:
                self.state.robot_state = RobotState.FOLLOWING
                self.state.last_vx     = vx
                self.state.last_vyaw   = vyaw

            await self.cmd.move(vx, vy, vyaw)
            log.debug("CMD → vx=%.3f vy=%.3f vyaw=%.3f | lidar=%.2f m",
                      vx, vy, vyaw, min_front_dist)

            # ── Mantener cadencia del loop ────────────────────────────────
            elapsed = time.time() - t0
            sleep_t = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_t)

    # ── Iniciar aplicación ───────────────────────────────────────────────

    async def run(self):
        self._running = True

        # Iniciar hilo de detección SSD
        det_thread = threading.Thread(
            target=self._detection_worker,
            name="SSD-Detector",
            daemon=True
        )
        det_thread.start()

        # Registrar callbacks en la conexión WebRTC
        # (los nombres exactos dependen de la versión de unitree_webrtc_connect)
        try:
            self.conn.video.add_frame_callback(self._on_video_frame)
            log.info("Callback de video registrado ✓")
        except AttributeError:
            log.warning("No se encontró conn.video — revisá la versión de la librería")

        try:
            self.conn.lidar.add_pointcloud_callback(self._on_lidar_data)
            log.info("Callback de LiDAR registrado ✓")
        except AttributeError:
            log.warning("No se encontró conn.lidar — el LiDAR no estará disponible")

        # Conectar al robot
        log.info("Conectando al Go2 en %s...", self.cfg.robot_ip)
        await self.conn.connect()
        log.info("Conexión establecida ✓")

        # Ejecutar control loop
        try:
            await self._control_loop()
        except asyncio.CancelledError:
            pass
        finally:
            log.info("Deteniendo robot...")
            await self.cmd.stop()
            self._running = False
            log.info("Robot detenido. Bye 🐕")


# ════════════════════════════════════════════════════════════════════════════
#  DEBUG / VISUALIZACIÓN (opcional)
# ════════════════════════════════════════════════════════════════════════════

class DebugVisualizer:
    """
    Muestra ventana OpenCV con overlay de detección y estado.
    Solo usar en desarrollo — añade latencia.
    """

    def __init__(self, app: RescueDogApp):
        self.app = app

    async def run(self):
        cfg = self.app.cfg
        cap = cv2.VideoCapture(0)   # solo para preview local si no hay acceso al stream
        interval = 1.0 / 10.0      # 10 fps de visualización

        while self.app._running:
            # Intentar obtener el último frame de la cola sin bloquearse
            try:
                frame = self.app._frame_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(interval)
                continue

            (bbox, fw, _, min_dist, state, vx, vyaw) = self.app.state.snapshot()

            # Overlay bbox persona
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "PERSONA", (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Centro imagen
            cv2.line(frame, (fw//2 - 80, frame.shape[0]//2),
                     (fw//2 + 80, frame.shape[0]//2), (255,0,0), 1)
            cv2.line(frame, (fw//2, 0), (fw//2, frame.shape[0]), (255,0,0), 1)

            # Info HUD
            color_state = {
                RobotState.FOLLOWING: (0,255,0),
                RobotState.SEARCHING: (0,165,255),
                RobotState.EMERGENCY: (0,0,255),
            }.get(state, (200,200,200))

            cv2.putText(frame, f"Estado: {state.value}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_state, 2)
            cv2.putText(frame, f"LiDAR: {min_dist:.2f}m", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"vx={vx:.2f}  vyaw={vyaw:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            cv2.imshow("Rescue Dog - Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.app._running = False
                break

            await asyncio.sleep(interval)

        cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Rescue Dog — Unitree Go2 autonomous person follower"
    )
    p.add_argument("--ip",              default="192.168.8.181",
                   help="IP del robot Go2 (modo STA)")
    p.add_argument("--method",          default="STA", choices=["AP", "STA", "Remote"],
                   help="Método de conexión")
    p.add_argument("--target-distance", type=float, default=1.2,
                   help="Distancia objetivo a la persona (metros)")
    p.add_argument("--max-speed",       type=float, default=0.5,
                   help="Velocidad máxima adelante (m/s)")
    p.add_argument("--confidence",      type=float, default=0.55,
                   help="Umbral de confianza del detector SSD (0-1)")
    p.add_argument("--debug",           action="store_true",
                   help="Mostrar ventana de debug con OpenCV")
    p.add_argument("--prototxt",        default="MobileNetSSD_deploy.prototxt")
    p.add_argument("--model",           default="MobileNetSSD_deploy.caffemodel")
    return p.parse_args()


async def main_async(cfg: Config, debug: bool):
    app = RescueDogApp(cfg)
    tasks = [asyncio.create_task(app.run())]

    if debug:
        vis = DebugVisualizer(app)
        tasks.append(asyncio.create_task(vis.run()))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        robot_ip          = args.ip,
        connection_method = args.method,
        target_distance   = args.target_distance,
        max_vx            = args.max_speed,
        ssd_confidence    = args.confidence,
        ssd_prototxt      = args.prototxt,
        ssd_model         = args.model,
    )

    print("""
╔══════════════════════════════════════════════════════╗
║  RESCUE DOG — Unitree Go2 Autonomous Person Follower ║
║  Presioná Ctrl+C para detener el robot               ║
╚══════════════════════════════════════════════════════╝
    """)

    asyncio.run(main_async(cfg, debug=args.debug))