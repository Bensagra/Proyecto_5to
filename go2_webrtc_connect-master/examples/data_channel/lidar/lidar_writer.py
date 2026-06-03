#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import csv
import inspect
import os
import signal
import sys
import threading
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

LIDAR_TOPIC = "rt/utlidar/voxel_map_compressed"
LIDAR_SWITCH_TOPIC = "rt/utlidar/switch"


class SharedMapWriter:
    def __init__(
        self,
        live_npy_path: str = "live_points.npy",
        final_csv_path: str = "mapa_final.csv",
        mode: str = "current",  # current | rolling | global
        rolling_frames: int = 4,
        voxel_size_frame: float = 0.02,
        voxel_size_map: float = 0.025,
        write_every_n_frames: int = 1,
        min_z: Optional[float] = -0.05,
        max_z: Optional[float] = 1.40,
        max_range: Optional[float] = 6.0,
    ):
        self.live_npy_path = live_npy_path
        self.final_csv_path = final_csv_path
        self.mode = mode
        self.rolling_frames = rolling_frames
        self.voxel_size_frame = voxel_size_frame
        self.voxel_size_map = voxel_size_map
        self.write_every_n_frames = write_every_n_frames
        self.min_z = min_z
        self.max_z = max_z
        self.max_range = max_range

        self.lock = threading.Lock()
        self.frame_counter = 0

        self.current_points = np.empty((0, 3), dtype=np.float32)
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.rolling_buffer = deque(maxlen=rolling_frames)

    def _voxel_downsample_numpy(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        if len(points) == 0 or voxel_size <= 0:
            return points
        coords = np.floor(points / voxel_size).astype(np.int32)
        _, idx = np.unique(coords, axis=0, return_index=True)
        idx = np.sort(idx)
        return points[idx]

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        mask = np.ones(len(points), dtype=bool)

        if self.min_z is not None:
            mask &= points[:, 2] >= self.min_z
        if self.max_z is not None:
            mask &= points[:, 2] <= self.max_z
        if self.max_range is not None:
            r = np.linalg.norm(points[:, :3], axis=1)
            mask &= r <= self.max_range

        return points[mask]

    def _write_live_npy_atomic(self, points: np.ndarray):
        final_path = self.live_npy_path
        if not final_path.endswith(".npy"):
            final_path += ".npy"

        tmp_path = final_path + ".tmp.npy"
        np.save(tmp_path, points.astype(np.float32))
        os.replace(tmp_path, final_path)

    def add_frame(self, points: np.ndarray):
        if points is None or len(points) == 0:
            return

        frame = points[:, :3].astype(np.float32)
        frame = self._filter_points(frame)
        frame = self._voxel_downsample_numpy(frame, self.voxel_size_frame)

        with self.lock:
            self.current_points = frame
            self.frame_counter += 1

            if self.mode == "current":
                out = self.current_points

            elif self.mode == "rolling":
                self.rolling_buffer.append(frame)
                if len(self.rolling_buffer) == 0:
                    out = np.empty((0, 3), dtype=np.float32)
                else:
                    merged = np.vstack(list(self.rolling_buffer))
                    out = self._voxel_downsample_numpy(merged, self.voxel_size_map)

            elif self.mode == "global":
                if len(self.global_points) == 0:
                    merged = frame
                else:
                    merged = np.vstack([self.global_points, frame])
                self.global_points = self._voxel_downsample_numpy(merged, self.voxel_size_map)
                out = self.global_points
            else:
                out = self.current_points

            if self.frame_counter % self.write_every_n_frames == 0:
                self._write_live_npy_atomic(out)

    def flush(self):
        with self.lock:
            if self.mode == "current":
                out = self.current_points
            elif self.mode == "rolling":
                if len(self.rolling_buffer) == 0:
                    out = np.empty((0, 3), dtype=np.float32)
                else:
                    out = self._voxel_downsample_numpy(
                        np.vstack(list(self.rolling_buffer)),
                        self.voxel_size_map,
                    )
            else:
                out = self.global_points

            self._write_live_npy_atomic(out)
            return out.copy()

    def save_final_csv(self):
        pts = self.flush()
        with open(self.final_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(pts.tolist())
        print(f"\n[OK] CSV final guardado en {self.final_csv_path}")


class Go2LidarWriter:
    def __init__(self, conn, mapper: SharedMapWriter):
        self.conn = conn
        self.mapper = mapper
        self.running = True
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
                return origin + vox * float(resolution)

            if "points" in payload:
                return self._extract_points(payload["points"])

            if "data" in payload and topic != LIDAR_TOPIC:
                return self._extract_points(payload["data"])

        if isinstance(payload, np.ndarray) and payload.ndim == 2 and payload.shape[1] >= 3:
            return payload[:, :3].astype(np.float32)

        return None

    def on_lidar_message(self, message: Any):
        points = self._extract_points(message)
        if points is None or len(points) == 0:
            return

        self.total_frames += 1
        self.mapper.add_frame(points)

        mins = points.min(axis=0)
        maxs = points.max(axis=0)

        sys.stdout.write(
            f"\rFrames: {self.total_frames:6d} | "
            f"FramePts: {len(points):7d} | "
            f"X[{mins[0]: .2f}, {maxs[0]: .2f}] "
            f"Y[{mins[1]: .2f}, {maxs[1]: .2f}] "
            f"Z[{mins[2]: .2f}, {maxs[2]: .2f}]      "
        )
        sys.stdout.flush()

    def enable_lidar_stream(self):
        candidates = [self._safe_get(self.conn, "datachannel"), self.conn]

        for obj in candidates:
            if obj is None:
                continue
            pubsub = self._safe_get(obj, "pub_sub") or self._safe_get(obj, "pubsub")
            if pubsub is not None:
                for name in ["publish_without_callback", "publish", "send"]:
                    method = self._safe_get(pubsub, name)
                    if callable(method):
                        for args in [
                            (LIDAR_SWITCH_TOPIC, "on"),
                            (LIDAR_SWITCH_TOPIC, {"data": "on"}),
                            (LIDAR_SWITCH_TOPIC, b"on"),
                        ]:
                            try:
                                method(*args)
                                return True
                            except Exception:
                                pass
        return False

    def _try_subscribe_on_object(self, obj: Any, topic: str, callback: Callable[[Any], None]) -> bool:
        if obj is None:
            return False

        for method_name in [
            "subscribe",
            "subscribe_to_topic",
            "add_subscription",
            "register_topic_callback",
            "on",
            "listen",
        ]:
            method = self._safe_get(obj, method_name)
            if not callable(method):
                continue

            for args in [
                (topic, callback),
                (callback, topic),
                (topic, callback, True),
            ]:
                try:
                    method(*args)
                    return True
                except Exception:
                    pass
        return False

    def install_lidar_subscription(self):
        objects_to_try = [
            self.conn,
            self._safe_get(self.conn, "datachannel"),
            self._safe_get(self._safe_get(self.conn, "datachannel"), "pub_sub"),
            self._safe_get(self._safe_get(self.conn, "datachannel"), "pubsub"),
            self._safe_get(self.conn, "pub_sub"),
            self._safe_get(self.conn, "pubsub"),
        ]

        for obj in objects_to_try:
            if self._try_subscribe_on_object(obj, LIDAR_TOPIC, self.on_lidar_message):
                return

        raise RuntimeError("No encontré método de suscripción compatible.")

    async def run(self):
        print("[INFO] Conectando al robot...")
        await self.conn.connect()
        print("[INFO] Conexión WebRTC establecida.")

        self.enable_lidar_stream()
        self.install_lidar_subscription()

        while self.running:
            await asyncio.sleep(0.03)

        self.mapper.save_final_csv()

        close_method = self._safe_get(self.conn, "disconnect") or self._safe_get(self.conn, "close")
        if callable(close_method):
            maybe = close_method()
            if inspect.isawaitable(maybe):
                await maybe


def build_connection(args):
    if args.mode == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    if args.mode == "sta":
        if args.ip:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        if args.serial:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial)
        raise ValueError("En modo sta pasá --ip o --serial")

    if args.mode == "remote":
        return UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )

    raise ValueError("Modo inválido")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ap", "sta", "remote"], required=True)
    p.add_argument("--ip")
    p.add_argument("--serial")
    p.add_argument("--username")
    p.add_argument("--password")
    p.add_argument("--live-npy", default="live_points.npy")
    p.add_argument("--final-csv", default="mapa_final.csv")
    p.add_argument("--map-mode", choices=["current", "rolling", "global"], default="current")
    p.add_argument("--rolling-frames", type=int, default=4)
    p.add_argument("--voxel-size-frame", type=float, default=0.02)
    p.add_argument("--voxel-size-map", type=float, default=0.025)
    p.add_argument("--write-every-n-frames", type=int, default=1)
    p.add_argument("--min-z", type=float, default=-0.05)
    p.add_argument("--max-z", type=float, default=1.40)
    p.add_argument("--max-range", type=float, default=6.0)
    return p.parse_args()


async def async_main():
    args = parse_args()
    conn = build_connection(args)

    mapper = SharedMapWriter(
        live_npy_path=args.live_npy,
        final_csv_path=args.final_csv,
        mode=args.map_mode,
        rolling_frames=args.rolling_frames,
        voxel_size_frame=args.voxel_size_frame,
        voxel_size_map=args.voxel_size_map,
        write_every_n_frames=args.write_every_n_frames,
        min_z=args.min_z,
        max_z=args.max_z,
        max_range=args.max_range,
    )

    app = Go2LidarWriter(conn, mapper)
    loop = asyncio.get_running_loop()

    def stop_handler():
        app.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_handler)
        except NotImplementedError:
            pass

    await app.run()


if __name__ == "__main__":
    asyncio.run(async_main())