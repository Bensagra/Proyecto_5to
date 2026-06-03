#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lee LiDAR del Unitree Go2 usando legion1581/unitree_webrtc_connect
adaptado al payload real: rt/utlidar/voxel_map_compressed

Instalación:
    pip install unitree_webrtc_connect numpy

Ejemplos:
    python read_go2_lidar.py --mode ap
    python read_go2_lidar.py --mode sta --ip 192.168.8.181
    python read_go2_lidar.py --mode sta --serial B42D2000XXXXXXXX
    python read_go2_lidar.py --mode remote --serial B42D2000XXXXXXXX --username tu_mail --password tu_pass
    python read_go2_lidar.py --mode sta --ip 192.168.8.181 --save-csv lidar.csv
"""

import argparse
import asyncio
import csv
import inspect
import signal
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

LIDAR_TOPIC = "rt/utlidar/voxel_map_compressed"
LIDAR_SWITCH_TOPIC = "rt/utlidar/switch"


class LidarReader:
    def __init__(self, conn: UnitreeWebRTCConnection, save_csv: Optional[str] = None):
        self.conn = conn
        self.save_csv = save_csv
        self.running = True
        self.frame_count = 0
        self.last_points = None
        self.last_raw = None
        self.last_ts = None

    # ------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------
    def _safe_get(self, obj: Any, attr: str, default=None):
        try:
            return getattr(obj, attr, default)
        except Exception:
            return default

    # ------------------------------------------------------------
    # Decodificación del payload real del Go2
    # ------------------------------------------------------------
    def _extract_points(self, payload: Any) -> Optional[np.ndarray]:
        """
        Convierte el mensaje rt/utlidar/voxel_map_compressed a Nx3 float32.

        Formato observado:
        {
            'type': 'msg',
            'topic': 'rt/utlidar/voxel_map_compressed',
            'data': {
                'stamp': ...,
                'frame_id': 'odom',
                'resolution': 0.05,
                'origin': [ox, oy, oz],
                'width': [128, 128, 38],
                'data': {
                    'point_count': ...,
                    'positions': np.array([...], dtype=uint8),
                    'uvs': ...,
                    'indices': ...
                }
            }
        }

        positions viene como [ix, iy, iz, ix, iy, iz, ...]
        y se reconstruye como:
            xyz = origin + positions * resolution
        """

        if payload is None:
            return None

        # Caso principal: dict del topic comprimido
        if isinstance(payload, dict):
            topic = payload.get("topic")

            if topic == LIDAR_TOPIC:
                outer = payload.get("data", {})
                origin = outer.get("origin")
                resolution = outer.get("resolution")
                inner = outer.get("data", {})
                positions = inner.get("positions")

                if origin is None or resolution is None or positions is None:
                    return None

                arr = np.asarray(positions, dtype=np.float32)

                if arr.ndim != 1 or len(arr) % 3 != 0:
                    return None

                vox = arr.reshape(-1, 3)
                origin = np.asarray(origin, dtype=np.float32).reshape(1, 3)
                points = origin + vox * float(resolution)
                return points

            # Fallbacks
            if "points" in payload:
                return self._extract_points(payload["points"])

            if "data" in payload and topic != LIDAR_TOPIC:
                return self._extract_points(payload["data"])

            # caso dict simple con x,y,z
            if all(k in payload for k in ("x", "y", "z")):
                try:
                    row = [float(payload["x"]), float(payload["y"]), float(payload["z"])]
                    return np.asarray([row], dtype=np.float32)
                except Exception:
                    return None

        # Caso numpy Nx3
        if isinstance(payload, np.ndarray):
            if payload.ndim == 2 and payload.shape[1] >= 3:
                return payload[:, :3].astype(np.float32)

        # Caso lista Nx3
        if isinstance(payload, list):
            try:
                arr = np.asarray(payload, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3]
            except Exception:
                pass

        return None

    # ------------------------------------------------------------
    # Callback de LiDAR
    # ------------------------------------------------------------
    def on_lidar_message(self, message: Any):
        self.last_raw = message
        self.last_ts = time.time()

        points = self._extract_points(message)
        if points is None:
            print("\n[LiDAR] Mensaje recibido pero no pude normalizarlo.")
            print(f"[LiDAR] Tipo de payload: {type(message)}")
            try:
                preview = str(message)
                if len(preview) > 700:
                    preview = preview[:700] + "..."
                print(f"[LiDAR] Preview: {preview}")
            except Exception:
                pass
            return

        self.last_points = points
        self.frame_count += 1

        xyz = points[:, :3]
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)

        sys.stdout.write(
            f"\rFrames: {self.frame_count:6d} | "
            f"Puntos: {len(points):6d} | "
            f"X[{mins[0]: .2f}, {maxs[0]: .2f}] "
            f"Y[{mins[1]: .2f}, {maxs[1]: .2f}] "
            f"Z[{mins[2]: .2f}, {maxs[2]: .2f}]      "
        )
        sys.stdout.flush()

    # ------------------------------------------------------------
    # Publicar switch LiDAR = on
    # ------------------------------------------------------------
    def enable_lidar_stream(self):
        """
        Intenta activar el stream LiDAR publicando 'on' en rt/utlidar/switch.
        """
        candidates = [
            self._safe_get(self.conn, "datachannel"),
            self.conn,
        ]

        for obj in candidates:
            if obj is None:
                continue

            pubsub = self._safe_get(obj, "pub_sub") or self._safe_get(obj, "pubsub")
            if pubsub is not None:
                methods = [
                    "publish_without_callback",
                    "publish",
                    "send",
                ]
                for name in methods:
                    method = self._safe_get(pubsub, name)
                    if callable(method):
                        for args in [
                            (LIDAR_SWITCH_TOPIC, "on"),
                            (LIDAR_SWITCH_TOPIC, {"data": "on"}),
                            (LIDAR_SWITCH_TOPIC, b"on"),
                        ]:
                            try:
                                method(*args)
                                print(f"\n[OK] LiDAR habilitado con {pubsub.__class__.__name__}.{name}{args}")
                                return True
                            except Exception:
                                continue

        print("\n[WARN] No pude publicar el switch de LiDAR. Sigo igual por si ya está activo.")
        return False

    # ------------------------------------------------------------
    # Suscripción robusta
    # ------------------------------------------------------------
    def _try_subscribe_on_object(self, obj: Any, topic: str, callback: Callable[[Any], None]) -> bool:
        if obj is None:
            return False

        candidate_methods = [
            "subscribe",
            "subscribe_to_topic",
            "add_subscription",
            "register_topic_callback",
            "on",
            "listen",
        ]

        for method_name in candidate_methods:
            method = self._safe_get(obj, method_name)
            if not callable(method):
                continue

            attempts = [
                (topic, callback),
                (callback, topic),
                (topic, callback, True),
            ]

            for args in attempts:
                try:
                    method(*args)
                    print(f"\n[OK] Suscripto usando {obj.__class__.__name__}.{method_name}{args}")
                    return True
                except TypeError:
                    continue
                except Exception:
                    continue

        return False

    def install_lidar_subscription(self):
        objects_to_try = [
            self.conn,
            self._safe_get(self.conn, "datachannel"),
            self._safe_get(self._safe_get(self.conn, "datachannel"), "pub_sub"),
            self._safe_get(self._safe_get(self.conn, "datachannel"), "pubsub"),
            self._safe_get(self._safe_get(self.conn, "datachannel"), "subscriber"),
            self._safe_get(self.conn, "pub_sub"),
            self._safe_get(self.conn, "pubsub"),
        ]

        for obj in objects_to_try:
            if self._try_subscribe_on_object(obj, LIDAR_TOPIC, self.on_lidar_message):
                return

        raise RuntimeError(
            "No encontré un método de suscripción compatible en esta versión de unitree_webrtc_connect."
        )

    # ------------------------------------------------------------
    # Guardado CSV
    # ------------------------------------------------------------
    def save_last_frame_to_csv(self, path: str):
        if self.last_points is None or len(self.last_points) == 0:
            print("\n[WARN] No hay frame LiDAR para guardar.")
            return

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(self.last_points.tolist())

        print(f"\n[OK] Último frame guardado en: {path}")

    # ------------------------------------------------------------
    # Debug opcional
    # ------------------------------------------------------------
    def print_first_points(self, count: int = 5):
        if self.last_points is None or len(self.last_points) == 0:
            return
        print("\nPrimeros puntos:")
        for row in self.last_points[:count]:
            print(f"  {row}")

    # ------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------
    async def run(self):
        print("[INFO] Conectando al robot...")
        await self.conn.connect()
        print("[INFO] Conexión WebRTC establecida.")

        self.enable_lidar_stream()
        self.install_lidar_subscription()

        print(f"[INFO] Esperando frames LiDAR en topic: {LIDAR_TOPIC}")

        printed_sample = False

        while self.running:
            if (not printed_sample) and self.last_points is not None and len(self.last_points) > 0:
                self.print_first_points()
                printed_sample = True

            await asyncio.sleep(0.1)

        if self.save_csv:
            self.save_last_frame_to_csv(self.save_csv)

        close_method = self._safe_get(self.conn, "disconnect") or self._safe_get(self.conn, "close")
        if callable(close_method):
            try:
                maybe = close_method()
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                pass

        print("\n[INFO] Fin.")

# ------------------------------------------------------------
# Conexión
# ------------------------------------------------------------
def build_connection(args) -> UnitreeWebRTCConnection:
    if args.mode == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    if args.mode == "sta":
        if args.ip:
            return UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=args.ip,
            )
        if args.serial:
            return UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                serialNumber=args.serial,
            )
        raise ValueError("En modo 'sta' tenés que pasar --ip o --serial")

    if args.mode == "remote":
        if not args.serial or not args.username or not args.password:
            raise ValueError("En modo 'remote' tenés que pasar --serial --username --password")
        return UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )

    raise ValueError(f"Modo no soportado: {args.mode}")

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Leer LiDAR del Unitree Go2")
    parser.add_argument("--mode", choices=["ap", "sta", "remote"], required=True, help="Modo de conexión")
    parser.add_argument("--ip", help="IP del robot para modo STA")
    parser.add_argument("--serial", help="Serial del robot para modo STA o REMOTE")
    parser.add_argument("--username", help="Usuario Unitree para modo REMOTE")
    parser.add_argument("--password", help="Password Unitree para modo REMOTE")
    parser.add_argument("--save-csv", help="Ruta CSV para guardar el último frame al salir", default=None)
    return parser.parse_args()

# ------------------------------------------------------------
# Main async
# ------------------------------------------------------------
async def async_main():
    args = parse_args()
    conn = build_connection(args)
    reader = LidarReader(conn, save_csv=args.save_csv)

    loop = asyncio.get_running_loop()

    def stop_handler():
        reader.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_handler)
        except NotImplementedError:
            pass

    await reader.run()

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por usuario.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

if __name__ == "__main__":
    main()