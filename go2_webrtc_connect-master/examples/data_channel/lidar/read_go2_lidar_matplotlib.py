#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mapa 3D en vivo del LiDAR del Unitree Go2 sin Open3D.
Usa matplotlib para visualizar en vivo y guarda el mapa en CSV.

Instalación:
    pip install unitree_webrtc_connect numpy matplotlib

Ejemplos:
    python go2_lidar_live_map_matplotlib.py --mode ap
    python go2_lidar_live_map_matplotlib.py --mode sta --ip 192.168.8.181
    python go2_lidar_live_map_matplotlib.py --mode sta --ip 192.168.8.181 --save-csv mapa_go2.csv
"""

import argparse
import asyncio
import csv
import inspect
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

LIDAR_TOPIC = "rt/utlidar/voxel_map_compressed"
LIDAR_SWITCH_TOPIC = "rt/utlidar/switch"


class LiveMap3DMatplotlib:
    def __init__(
        self,
        voxel_size_map: float = 0.05,
        voxel_size_frame: float = 0.05,
        update_vis_every_n_frames: int = 2,
        max_points_before_merge: int = 120000,
        vis_stride: int = 8,
        save_csv: Optional[str] = "go2_map.csv",
    ):
        self.voxel_size_map = voxel_size_map
        self.voxel_size_frame = voxel_size_frame
        self.update_vis_every_n_frames = update_vis_every_n_frames
        self.max_points_before_merge = max_points_before_merge
        self.vis_stride = max(1, vis_stride)
        self.save_csv = save_csv

        self.lock = threading.Lock()

        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.pending_points = np.empty((0, 3), dtype=np.float32)

        self.frame_counter = 0
        self.should_close = False

        self.fig = None
        self.ax = None
        self.scatter = None

    def _voxel_downsample_numpy(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        if len(points) == 0:
            return points

        coords = np.floor(points / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(coords, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return points[unique_idx]

    def add_frame(self, points: np.ndarray):
        if points is None or len(points) == 0:
            return

        frame = points[:, :3].astype(np.float32)

        if self.voxel_size_frame and self.voxel_size_frame > 0:
            frame = self._voxel_downsample_numpy(frame, self.voxel_size_frame)

        with self.lock:
            if len(self.pending_points) == 0:
                self.pending_points = frame
            else:
                self.pending_points = np.vstack([self.pending_points, frame])

            self.frame_counter += 1

            need_merge = (
                self.frame_counter % self.update_vis_every_n_frames == 0
                or len(self.pending_points) >= self.max_points_before_merge
            )

            if need_merge:
                if len(self.global_points) == 0:
                    merged = self.pending_points
                else:
                    merged = np.vstack([self.global_points, self.pending_points])

                if self.voxel_size_map and self.voxel_size_map > 0:
                    merged = self._voxel_downsample_numpy(merged, self.voxel_size_map)

                self.global_points = merged
                self.pending_points = np.empty((0, 3), dtype=np.float32)

    def flush_pending(self):
        with self.lock:
            if len(self.pending_points) == 0:
                return

            if len(self.global_points) == 0:
                merged = self.pending_points
            else:
                merged = np.vstack([self.global_points, self.pending_points])

            if self.voxel_size_map and self.voxel_size_map > 0:
                merged = self._voxel_downsample_numpy(merged, self.voxel_size_map)

            self.global_points = merged
            self.pending_points = np.empty((0, 3), dtype=np.float32)

    def start_visualizer(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("Go2 LiDAR Live Map")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.scatter = None

    def update_visualizer(self):
        if self.fig is None or self.ax is None:
            return

        with self.lock:
            pts = self.global_points.copy()

        if len(pts) == 0:
            plt.pause(0.001)
            return

        pts_vis = pts[::self.vis_stride]

        self.ax.cla()
        self.ax.set_title(f"Go2 LiDAR Live Map | puntos: {len(pts)}")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        x = pts_vis[:, 0]
        y = pts_vis[:, 1]
        z = pts_vis[:, 2]

        self.ax.scatter(x, y, z, s=1)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        self.ax.set_xlim(mins[0], maxs[0])
        self.ax.set_ylim(mins[1], maxs[1])
        self.ax.set_zlim(mins[2], maxs[2])

        plt.pause(0.001)

    def save_outputs(self):
        self.flush_pending()

        with self.lock:
            pts = self.global_points.copy()

        if len(pts) == 0:
            print("\n[WARN] No hay puntos para guardar.")
            return

        if self.save_csv:
            with open(self.save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                writer.writerows(pts.tolist())
            print(f"\n[OK] Mapa guardado en CSV: {self.save_csv}")

    def stop(self):
        self.should_close = True
        try:
            plt.ioff()
            plt.close("all")
        except Exception:
            pass


class Go2LidarMapper:
    def __init__(self, conn: UnitreeWebRTCConnection, mapper: LiveMap3DMatplotlib):
        self.conn = conn
        self.mapper = mapper
        self.running = True
        self.last_points = None
        self.last_ts = None
        self.total_frames = 0

    def _safe_get(self, obj: Any, attr: str, default=None):
        try:
            return getattr(obj, attr, default)
        except Exception:
            return default

    def _extract_points(self, payload: Any) -> Optional[np.ndarray]:
        if payload is None:
            return None

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

            if "points" in payload:
                return self._extract_points(payload["points"])

            if "data" in payload and topic != LIDAR_TOPIC:
                return self._extract_points(payload["data"])

        if isinstance(payload, np.ndarray):
            if payload.ndim == 2 and payload.shape[1] >= 3:
                return payload[:, :3].astype(np.float32)

        if isinstance(payload, list):
            try:
                arr = np.asarray(payload, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3]
            except Exception:
                pass

        return None

    def on_lidar_message(self, message: Any):
        self.last_ts = time.time()

        points = self._extract_points(message)
        if points is None:
            print("\n[LiDAR] Mensaje recibido pero no pude normalizarlo.")
            print(f"[LiDAR] Tipo de payload: {type(message)}")
            return

        self.last_points = points
        self.total_frames += 1
        self.mapper.add_frame(points)

        xyz = points[:, :3]
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)

        with self.mapper.lock:
            total_map_points = len(self.mapper.global_points) + len(self.mapper.pending_points)

        sys.stdout.write(
            f"\rFrames: {self.total_frames:6d} | "
            f"FramePts: {len(points):7d} | "
            f"MapPts: {total_map_points:7d} | "
            f"X[{mins[0]: .2f}, {maxs[0]: .2f}] "
            f"Y[{mins[1]: .2f}, {maxs[1]: .2f}] "
            f"Z[{mins[2]: .2f}, {maxs[2]: .2f}]      "
        )
        sys.stdout.flush()

    def enable_lidar_stream(self):
        candidates = [
            self._safe_get(self.conn, "datachannel"),
            self.conn,
        ]

        for obj in candidates:
            if obj is None:
                continue

            pubsub = self._safe_get(obj, "pub_sub") or self._safe_get(obj, "pubsub")
            if pubsub is not None:
                methods = ["publish_without_callback", "publish", "send"]
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

        raise RuntimeError("No encontré un método de suscripción compatible en esta versión.")

    async def run(self):
        print("[INFO] Conectando al robot...")
        await self.conn.connect()
        print("[INFO] Conexión WebRTC establecida.")

        self.mapper.start_visualizer()

        self.enable_lidar_stream()
        self.install_lidar_subscription()

        print(f"[INFO] Esperando frames LiDAR en topic: {LIDAR_TOPIC}")
        print("[INFO] Visualización iniciada con Matplotlib.")

        vis_counter = 0

        while self.running:
            vis_counter += 1
            if vis_counter % 4 == 0:
                self.mapper.update_visualizer()
            await asyncio.sleep(0.05)

        print("\n[INFO] Cerrando y guardando mapa...")
        self.mapper.save_outputs()
        self.mapper.stop()

        close_method = self._safe_get(self.conn, "disconnect") or self._safe_get(self.conn, "close")
        if callable(close_method):
            try:
                maybe = close_method()
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                pass

        print("[INFO] Fin.")


def build_connection(args) -> UnitreeWebRTCConnection:
    if args.mode == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    if args.mode == "sta":
        if args.ip:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        if args.serial:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Mapa 3D en vivo con LiDAR del Go2 usando Matplotlib")
    parser.add_argument("--mode", choices=["ap", "sta", "remote"], required=True)
    parser.add_argument("--ip")
    parser.add_argument("--serial")
    parser.add_argument("--username")
    parser.add_argument("--password")

    parser.add_argument("--save-csv", default="go2_map.csv")
    parser.add_argument("--voxel-size-map", type=float, default=0.05)
    parser.add_argument("--voxel-size-frame", type=float, default=0.05)
    parser.add_argument("--update-vis-every-n-frames", type=int, default=2)
    parser.add_argument("--max-points-before-merge", type=int, default=120000)
    parser.add_argument("--vis-stride", type=int, default=8, help="Muestra 1 de cada N puntos")

    return parser.parse_args()


async def async_main():
    args = parse_args()

    mapper = LiveMap3DMatplotlib(
        voxel_size_map=args.voxel_size_map,
        voxel_size_frame=args.voxel_size_frame,
        update_vis_every_n_frames=args.update_vis_every_n_frames,
        max_points_before_merge=args.max_points_before_merge,
        vis_stride=args.vis_stride,
        save_csv=args.save_csv,
    )

    conn = build_connection(args)
    app = Go2LidarMapper(conn, mapper)

    loop = asyncio.get_running_loop()

    def stop_handler():
        app.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_handler)
        except NotImplementedError:
            pass

    await app.run()


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