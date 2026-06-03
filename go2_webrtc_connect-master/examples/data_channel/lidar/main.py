import asyncio
import math
import time
from dataclasses import dataclass
from typing import Optional, Any

import cv2
import numpy as np

from movment_controller import Go2Controller, Go2Config, MotionCommand
from unitree_webrtc_connect.constants import RTC_TOPIC


# =========================================================
# CONFIG
# =========================================================

@dataclass
class BallSeekConfig:
    # HSV base para pelota de tenis
    hsv_lower: tuple = (22, 70, 70)
    hsv_upper: tuple = (48, 255, 255)

    # Un segundo rango opcional por si tu cámara cambia tonos
    hsv_lower_2: tuple = (18, 50, 50)
    hsv_upper_2: tuple = (60, 255, 255)

    # Detección
    min_area: int = 180
    min_radius_px: float = 6.0
    min_circularity: float = 0.45
    min_score_to_accept: float = 0.22

    # Hough circles
    use_hough: bool = True
    hough_dp: float = 1.2
    hough_min_dist: int = 30
    hough_param1: int = 120
    hough_param2: int = 18
    hough_min_radius: int = 5
    hough_max_radius: int = 120

    # Control
    target_radius_px: float = 85.0
    center_tolerance: float = 0.08
    strong_turn_tolerance: float = 0.20

    # Más rápido
    search_turn_speed_fast: float = 0.95
    search_turn_speed_slow: float = 0.65
    align_turn_speed_fast: float = 0.85
    align_turn_speed_slow: float = 0.45
    forward_speed_fast: float = 0.42
    forward_speed_slow: float = 0.22

    # Búsqueda activa
    search_forward_burst_speed: float = 0.22
    search_forward_burst_every_s: float = 2.2
    search_forward_burst_duration_s: float = 0.45
    search_direction_change_s: float = 2.8

    # Timing
    control_interval: float = 0.09
    lost_ball_timeout_s: float = 0.8
    startup_video_wait_s: float = 0.6

    # Obstáculos
    obstacle_stop_distance_m: float = 0.55
    obstacle_slow_distance_m: float = 0.90
    obstacle_turn_bias: float = 1.15

    # UI
    show_debug_window: bool = True
    debug_window_name: str = "Go2 - Tennis Ball Seek SAFE"

    # Smoothing
    ema_alpha: float = 0.45


class TennisBallSeekerSafe:
    def __init__(self, controller: Go2Controller, config: Optional[BallSeekConfig] = None):
        self.controller = controller
        self.config = config or BallSeekConfig()

        self.running = False

        self.latest_frame = None
        self.latest_frame_time = 0.0

        self.frame_width = None
        self.frame_height = None

        # pelota
        self.ball_found = False
        self.ball_x = None
        self.ball_y = None
        self.ball_radius = 0.0
        self.ball_score = 0.0
        self.last_seen_time = 0.0

        # smoothing
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_radius = None

        # búsqueda
        self.search_dir = 1.0
        self.last_search_flip = time.time()
        self.last_search_burst = time.time()

        # obstáculos
        self.last_obstacle_update = 0.0
        self.front_obstacle_distance = float("inf")
        self.left_obstacle_distance = float("inf")
        self.right_obstacle_distance = float("inf")
        self.obstacle_source_ok = False

        self._vision_task = None
        self._control_task = None

    # =========================================================
    # VIDEO
    # =========================================================

    async def on_video_track(self, track):
        print("[VIDEO] Track de video recibido.")
        while self.running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                self.latest_frame = img
                self.latest_frame_time = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[VIDEO] Error recibiendo frame: {e}")
                await asyncio.sleep(0.03)

    async def enable_video(self):
        if not self.controller.conn:
            raise RuntimeError("El robot no está conectado.")

        video = getattr(self.controller.conn, "video", None)
        if video is None:
            raise RuntimeError("La conexión no expone video.")

        video.add_track_callback(self.on_video_track)

        result = video.switchVideoChannel(True)
        if asyncio.iscoroutine(result):
            await result

        await asyncio.sleep(self.config.startup_video_wait_s)

    async def disable_video(self):
        if not self.controller.conn:
            return

        try:
            video = getattr(self.controller.conn, "video", None)
            if video is None:
                return
            result = video.switchVideoChannel(False)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"[VIDEO] No se pudo desactivar video: {e}")

    # =========================================================
    # SPORT STATE / OBSTÁCULOS
    # =========================================================

    def _safe_float(self, v: Any, default=float("inf")) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _extract_min_dist(self, obj: Any) -> float:
        """
        Intenta sacar una distancia mínima desde formatos varios.
        """
        if obj is None:
            return float("inf")

        if isinstance(obj, (int, float)):
            return self._safe_float(obj)

        if isinstance(obj, dict):
            vals = []
            for v in obj.values():
                d = self._extract_min_dist(v)
                if math.isfinite(d):
                    vals.append(d)
            return min(vals) if vals else float("inf")

        if isinstance(obj, (list, tuple)):
            vals = []
            for v in obj:
                d = self._extract_min_dist(v)
                if math.isfinite(d):
                    vals.append(d)
            return min(vals) if vals else float("inf")

        return float("inf")

    def _extract_lr_front(self, range_obstacle: Any):
        """
        Intenta interpretar frontal / izquierda / derecha de forma flexible.
        Si no encuentra estructura clara, usa el mínimo global como frontal.
        """
        front = float("inf")
        left = float("inf")
        right = float("inf")

        if isinstance(range_obstacle, dict):
            # casos comunes posibles
            for k, v in range_obstacle.items():
                kl = str(k).lower()
                d = self._extract_min_dist(v)

                if "front" in kl or "forward" in kl or kl in ("0", "center", "mid"):
                    front = min(front, d)
                elif "left" in kl or kl in ("l", "1"):
                    left = min(left, d)
                elif "right" in kl or kl in ("r", "2"):
                    right = min(right, d)

        # fallback si vino raro
        global_min = self._extract_min_dist(range_obstacle)
        if not math.isfinite(front):
            front = global_min

        if not math.isfinite(left):
            left = global_min

        if not math.isfinite(right):
            right = global_min

        return front, left, right

    def _sport_state_callback(self, message):
        """
        Callback tolerante a distintas estructuras del mensaje.
        """
        try:
            data = message
            if isinstance(message, dict) and "data" in message:
                data = message["data"]

            range_obstacle = None

            if isinstance(data, dict):
                if "range_obstacle" in data:
                    range_obstacle = data.get("range_obstacle")
                elif "data" in data and isinstance(data["data"], dict):
                    range_obstacle = data["data"].get("range_obstacle")

            if range_obstacle is None:
                return

            front, left, right = self._extract_lr_front(range_obstacle)

            self.front_obstacle_distance = front
            self.left_obstacle_distance = left
            self.right_obstacle_distance = right
            self.last_obstacle_update = time.time()
            self.obstacle_source_ok = True

        except Exception:
            # no frenamos el programa por una variación del payload
            pass

    async def subscribe_obstacles(self):
        """
        Intenta subscribirse al state channel.
        Según la versión, el método puede variar.
        """
        if not self.controller.conn:
            return

        pubsub = getattr(self.controller.conn, "datachannel", None)
        if pubsub is None:
            return

        pubsub = getattr(pubsub, "pub_sub", None)
        if pubsub is None:
            return

        topic = RTC_TOPIC.get("LF_SPORT_MOD_STATE")
        if topic is None:
            print("[OBS] RTC_TOPIC['LF_SPORT_MOD_STATE'] no encontrado.")
            return

        # Distintas firmas posibles según versión
        tried = []

        try:
            tried.append("subscribe(topic, cb)")
            result = pubsub.subscribe(topic, self._sport_state_callback)
            if asyncio.iscoroutine(result):
                await result
            print("[OBS] Suscripción a obstáculos activa.")
            return
        except Exception:
            pass

        try:
            tried.append("subscribe_topic(topic, cb)")
            result = pubsub.subscribe_topic(topic, self._sport_state_callback)
            if asyncio.iscoroutine(result):
                await result
            print("[OBS] Suscripción a obstáculos activa.")
            return
        except Exception:
            pass

        try:
            tried.append("subscribe(topic=..., callback=...)")
            result = pubsub.subscribe(topic=topic, callback=self._sport_state_callback)
            if asyncio.iscoroutine(result):
                await result
            print("[OBS] Suscripción a obstáculos activa.")
            return
        except Exception:
            pass

        print(f"[OBS] No pude subscribirme al state channel. Intentos: {tried}")

    def obstacle_too_close_front(self) -> bool:
        return self.front_obstacle_distance <= self.config.obstacle_stop_distance_m

    def obstacle_should_slow(self) -> bool:
        return self.front_obstacle_distance <= self.config.obstacle_slow_distance_m

    def preferred_turn_sign(self) -> float:
        # gira hacia el lado más libre
        if self.left_obstacle_distance > self.right_obstacle_distance:
            return 1.0   # izquierda
        return -1.0      # derecha

    # =========================================================
    # DETECCIÓN
    # =========================================================

    def _mask_ball(self, hsv):
        lower1 = np.array(self.config.hsv_lower, dtype=np.uint8)
        upper1 = np.array(self.config.hsv_upper, dtype=np.uint8)
        lower2 = np.array(self.config.hsv_lower_2, dtype=np.uint8)
        upper2 = np.array(self.config.hsv_upper_2, dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _score_candidate(self, hsv, mask, cx, cy, radius, contour=None):
        h, w = mask.shape[:2]
        if radius < self.config.min_radius_px:
            return 0.0

        area_circle = math.pi * radius * radius
        area_ratio = min(area_circle / max(h * w, 1), 1.0)

        circ_score = 0.6
        if contour is not None:
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            if peri > 0:
                circularity = 4.0 * math.pi * area / (peri * peri)
                circ_score = max(0.0, min(1.0, circularity))

        # fracción de píxeles válidos dentro del círculo
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (int(cx), int(cy)), int(radius), 255, -1)
        overlap = cv2.bitwise_and(mask, circle_mask)
        valid_ratio = float(np.count_nonzero(overlap)) / max(float(np.count_nonzero(circle_mask)), 1.0)
        valid_ratio = max(0.0, min(1.0, valid_ratio))

        # premio por cercanía al centro
        center_x = w / 2.0
        center_y = h / 2.0
        dx = abs(cx - center_x) / center_x
        dy = abs(cy - center_y) / center_y
        center_bonus = max(0.0, 1.0 - 0.5 * dx - 0.3 * dy)

        # pelota muy arriba suele ser falso positivo
        vertical_bonus = 1.0 if cy > h * 0.18 else 0.7

        size_bonus = min(radius / 60.0, 1.0)
        score = (
            0.35 * valid_ratio +
            0.25 * circ_score +
            0.20 * center_bonus +
            0.12 * size_bonus +
            0.08 * min(area_ratio * 25.0, 1.0)
        ) * vertical_bonus
        return float(score)

    def detect_ball(self, frame):
        debug = frame.copy()
        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h

        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = self._mask_ball(hsv)

        best = None
        best_score = 0.0

        # método 1: contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            circularity = 4.0 * math.pi * area / (peri * peri)
            if circularity < self.config.min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.config.min_radius_px:
                continue

            score = self._score_candidate(hsv, mask, x, y, radius, contour=cnt)
            if score > best_score:
                best_score = score
                best = (float(x), float(y), float(radius), score, "contour")

        # método 2: HoughCircles
        if self.config.use_hough:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=self.config.hough_dp,
                minDist=self.config.hough_min_dist,
                param1=self.config.hough_param1,
                param2=self.config.hough_param2,
                minRadius=self.config.hough_min_radius,
                maxRadius=self.config.hough_max_radius,
            )
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    score = self._score_candidate(hsv, mask, x, y, r, contour=None)
                    score *= 1.06
                    if score > best_score:
                        best_score = score
                        best = (float(x), float(y), float(r), score, "hough")

        cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(debug, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

        if best is None or best_score < self.config.min_score_to_accept:
            cv2.putText(
                debug,
                "Pelota: NO",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            return False, None, None, 0.0, 0.0, debug, mask

        cx, cy, radius, score, source = best
        cv2.circle(debug, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.circle(debug, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"Pelota SI r={radius:.1f} score={score:.2f} src={source}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        return True, cx, cy, radius, score, debug, mask

    def _ema(self, old, new):
        if old is None:
            return new
        a = self.config.ema_alpha
        return a * new + (1.0 - a) * old

    async def vision_loop(self):
        while self.running:
            frame = self.latest_frame

            if frame is None:
                await asyncio.sleep(0.02)
                continue

            found, cx, cy, radius, score, debug, mask = self.detect_ball(frame)

            if found:
                self.smooth_x = self._ema(self.smooth_x, cx)
                self.smooth_y = self._ema(self.smooth_y, cy)
                self.smooth_radius = self._ema(self.smooth_radius, radius)

                self.ball_found = True
                self.ball_x = self.smooth_x
                self.ball_y = self.smooth_y
                self.ball_radius = self.smooth_radius
                self.ball_score = score
                self.last_seen_time = time.time()
            else:
                self.ball_found = False
                self.ball_score = 0.0

            if self.config.show_debug_window:
                dbg = debug

                obs_txt = (
                    f"OBS front={self.front_obstacle_distance:.2f} "
                    f"left={self.left_obstacle_distance:.2f} right={self.right_obstacle_distance:.2f}"
                )
                cv2.putText(
                    dbg,
                    obs_txt,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(self.config.debug_window_name, dbg)
                cv2.imshow(self.config.debug_window_name + " MASK", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] ESC presionado. Deteniendo...")
                    self.running = False
                    break

            await asyncio.sleep(0.01)

    # =========================================================
    # CONTROL
    # =========================================================

    async def send_velocity(self, x=0.0, y=0.0, z=0.0):
        await self.controller.send_move(MotionCommand(x=x, y=y, z=z))

    async def active_search_motion(self):
        """
        Busca girando rápido y cada tanto mete un pequeño avance en arco.
        """
        now = time.time()

        if now - self.last_search_flip > self.config.search_direction_change_s:
            self.search_dir *= -1.0
            self.last_search_flip = now

        # Si adelante hay obstáculo, no avanza: solo gira hacia el lado más libre
        if self.obstacle_too_close_front():
            turn_sign = self.preferred_turn_sign()
            await self.send_velocity(
                x=0.0,
                y=0.0,
                z=turn_sign * self.config.search_turn_speed_fast * self.config.obstacle_turn_bias,
            )
            await asyncio.sleep(self.config.control_interval)
            return

        # Cada tanto, un pequeño avance en arco para explorar mejor
        if now - self.last_search_burst > self.config.search_forward_burst_every_s:
            self.last_search_burst = now
            t_end = now + self.config.search_forward_burst_duration_s

            while self.running and time.time() < t_end:
                if self.obstacle_too_close_front():
                    break
                await self.send_velocity(
                    x=self.config.search_forward_burst_speed,
                    y=0.0,
                    z=self.search_dir * self.config.search_turn_speed_slow,
                )
                await asyncio.sleep(self.config.control_interval)
            return

        await self.send_velocity(
            x=0.0,
            y=0.0,
            z=self.search_dir * self.config.search_turn_speed_fast,
        )
        await asyncio.sleep(self.config.control_interval)

    async def avoid_obstacle(self):
        turn_sign = self.preferred_turn_sign()
        await self.send_velocity(
            x=0.0,
            y=0.0,
            z=turn_sign * self.config.search_turn_speed_fast * self.config.obstacle_turn_bias,
        )
        await asyncio.sleep(self.config.control_interval)

    async def control_loop(self):
        while self.running:
            now = time.time()
            recently_seen = (now - self.last_seen_time) <= self.config.lost_ball_timeout_s

            # Seguridad primero
            if self.obstacle_too_close_front():
                await self.avoid_obstacle()
                continue

            # Si no ve pelota, búsqueda activa
            if not recently_seen:
                await self.active_search_motion()
                continue

            if self.ball_x is None or self.frame_width is None or self.ball_radius is None:
                await self.controller.stop()
                await asyncio.sleep(self.config.control_interval)
                continue

            frame_center_x = self.frame_width / 2.0
            error_x = (self.ball_x - frame_center_x) / frame_center_x
            radius = self.ball_radius

            # Si llegó cerca
            if radius >= self.config.target_radius_px:
                await self.controller.stop()
                print("[INFO] Pelota alcanzada o suficientemente cerca.")
                await asyncio.sleep(self.config.control_interval)
                continue

            # Si obstáculo moderado, bajar velocidad
            slow_mode = self.obstacle_should_slow()

            # Giro fuerte
            if abs(error_x) > self.config.strong_turn_tolerance:
                z = -self.config.align_turn_speed_fast if error_x > 0 else self.config.align_turn_speed_fast
                await self.send_velocity(x=0.0, y=0.0, z=z)
                await asyncio.sleep(self.config.control_interval)
                continue

            # Giro suave con avance en arco
            if abs(error_x) > self.config.center_tolerance:
                z = -self.config.align_turn_speed_slow if error_x > 0 else self.config.align_turn_speed_slow
                x = self.config.forward_speed_slow if not slow_mode else 0.0
                await self.send_velocity(x=x, y=0.0, z=z)
                await asyncio.sleep(self.config.control_interval)
                continue

            # Centrada: avanzar
            x = self.config.forward_speed_fast if not slow_mode else self.config.forward_speed_slow
            await self.send_velocity(x=x, y=0.0, z=0.0)
            await asyncio.sleep(self.config.control_interval)

        await self.controller.stop()

    # =========================================================
    # RUN
    # =========================================================

    async def run(self):
        self.running = True
        self.last_seen_time = 0.0
        self.last_search_flip = time.time()
        self.last_search_burst = time.time()

        await self.subscribe_obstacles()
        await self.enable_video()

        self._vision_task = asyncio.create_task(self.vision_loop(), name="vision_loop")
        self._control_task = asyncio.create_task(self.control_loop(), name="control_loop")

        try:
            await asyncio.gather(self._vision_task, self._control_task)
        finally:
            self.running = False

            for task in (self._vision_task, self._control_task):
                if task and not task.done():
                    task.cancel()

            await self.controller.stop()
            await self.disable_video()

            if self.config.show_debug_window:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass


# =========================================================
# MAIN
# =========================================================

async def main():
    controller = Go2Controller(
        Go2Config(
            connection_mode="LOCAL_AP",
            robot_ip="192.168.8.181",
            linear_speed_mps=0.5,
            angular_speed_rad_s=0.8,
        )
    )

    seeker = TennisBallSeekerSafe(
        controller,
        BallSeekConfig(
            hsv_lower=(22, 70, 70),
            hsv_upper=(48, 255, 255),
            hsv_lower_2=(18, 50, 50),
            hsv_upper_2=(60, 255, 255),
            min_area=180,
            min_radius_px=6.0,
            min_circularity=0.45,
            min_score_to_accept=0.22,
            use_hough=True,
            hough_dp=1.2,
            hough_min_dist=30,
            hough_param1=120,
            hough_param2=18,
            hough_min_radius=5,
            hough_max_radius=120,
            target_radius_px=85.0,
            center_tolerance=0.08,
            strong_turn_tolerance=0.20,
            search_turn_speed_fast=0.95,
            search_turn_speed_slow=0.65,
            align_turn_speed_fast=0.85,
            align_turn_speed_slow=0.45,
            forward_speed_fast=0.42,
            forward_speed_slow=0.22,
            search_forward_burst_speed=0.22,
            search_forward_burst_every_s=2.2,
            search_forward_burst_duration_s=0.45,
            search_direction_change_s=2.8,
            control_interval=0.09,
            lost_ball_timeout_s=0.8,
            obstacle_stop_distance_m=0.55,
            obstacle_slow_distance_m=0.90,
            obstacle_turn_bias=1.15,
            show_debug_window=True,
            ema_alpha=0.45,
        ),
    )

    await controller.connect()

    try:
        print("[INFO] Búsqueda autónoma de pelota de tenis iniciada.")
        print("[INFO] ESC para salir.")
        await seeker.run()
    finally:
        await controller.disconnect()


if __name__ == "__main__":
    asyncio.run(main())