#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRODUCER - Pipeline de Mapeo Texturizado para Unitree Go2.
LiDAR Voxel Stream + Pose (sportmodestate) + Cámara Frontal WebRTC.

Arquitectura igual a display_video_channel.py (que conecta perfecto):
  - asyncio corre en un thread dedicado
  - main thread procesa LiDAR + texturizado + escritura
  - una sola UnitreeWebRTCConnection sirve para video y data channel
"""

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Optional, Tuple

# Prefer the local checkout of legion1581/unitree_webrtc_connect when this
# project is run from the workspace, even if another version is installed.
LOCAL_UNITREE_ROOT = Path(__file__).resolve().parents[4] / "unitree_webrtc_connect"
if (LOCAL_UNITREE_ROOT / "unitree_webrtc_connect" / "webrtc_driver.py").exists():
    sys.path.insert(0, str(LOCAL_UNITREE_ROOT))

import cv2
import numpy as np
from aiortc import MediaStreamTrack

from unitree_webrtc_connect.constants import RTC_TOPIC
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

logging.basicConfig(level=logging.FATAL)
LOG = logging.getLogger("producer")

LIDAR_TOPIC = RTC_TOPIC["ULIDAR_ARRAY"]
LIDAR_SWITCH_TOPIC = RTC_TOPIC["ULIDAR_SWITCH"]
POSE_TOPIC_PRIMARY = RTC_TOPIC["LF_SPORT_MOD_STATE"]
POSE_TOPIC_FALLBACK = RTC_TOPIC["SPORT_MOD_STATE"]

DEFAULT_INTRINSICS = dict(fx=420.0, fy=420.0, cx=640.0, cy=360.0, width=1280, height=720)
DEFAULT_CAM_OFFSET_XYZ = (0.04, 0.0, -0.08)

R_CAM_FROM_LIDAR = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)

NEUTRAL_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float32)


# =========================================================
# MODELOS MATEMÁTICOS Y PROYECCIÓN
# =========================================================

@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    cam_offset_in_lidar: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_CAM_OFFSET_XYZ, dtype=np.float64)
    )
    R_cam_lidar: np.ndarray = field(default_factory=lambda: R_CAM_FROM_LIDAR.copy())
    min_depth: float = 0.10
    max_depth: float = 12.0

    def project_lidar_points(self, pts_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pts_lidar.size == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

        p_cam = (self.R_cam_lidar @ (pts_lidar - self.cam_offset_in_lidar).T).T
        z = p_cam[:, 2]
        depth_ok = (z > self.min_depth) & (z < self.max_depth)

        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * (p_cam[:, 0] / z) + self.cx
            v = self.fy * (p_cam[:, 1] / z) + self.cy

        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        in_image = (
            depth_ok
            & (u_int >= 0) & (u_int < self.width)
            & (v_int >= 0) & (v_int < self.height)
        )
        uv = np.stack([u_int, v_int], axis=1)
        uv[~in_image] = 0
        return uv, in_image


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


@dataclass
class Pose:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    stamp: float = 0.0

    def world_T_lidar(self, lidar_mount_offset_xyz=(0.0, 0.0, 0.30)) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rpy_to_R(*self.rpy)
        T[:3, 3] = self.position
        T_mount = np.eye(4, dtype=np.float64)
        T_mount[:3, 3] = list(lidar_mount_offset_xyz)
        return T @ T_mount


class PoseTracker:
    def __init__(self, lidar_mount_offset_xyz=(0.0, 0.0, 0.30)):
        self.lidar_mount = tuple(lidar_mount_offset_xyz)
        self._lock = threading.Lock()
        self._pose = Pose(stamp=time.time())
        self.have_real_pose = False
        self.trajectory = []

    def update(self, position_xyz, rpy_xyz):
        if len(position_xyz) < 3 or len(rpy_xyz) < 3:
            return
        now = time.time()
        with self._lock:
            self._pose = Pose(
                position=np.array(position_xyz[:3], dtype=np.float64),
                rpy=np.array(rpy_xyz[:3], dtype=np.float64),
                stamp=now,
            )
            self.have_real_pose = True
            self.trajectory.append((
                now,
                float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]),
                float(rpy_xyz[0]), float(rpy_xyz[1]), float(rpy_xyz[2]),
            ))

    def snapshot(self) -> Pose:
        with self._lock:
            return Pose(
                position=self._pose.position.copy(),
                rpy=self._pose.rpy.copy(),
                stamp=self._pose.stamp,
            )


# =========================================================
# BUFFERS Y ESTRUCTURAS DE ALMACENAMIENTO
# =========================================================

class VideoBuffer:
    def __init__(self, max_age_sec: float = 0.6):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stamp: float = 0.0
        self.max_age_sec = max_age_sec
        self.received_count = 0

    def push(self, bgr: np.ndarray):
        with self._lock:
            self._frame = bgr
            self._stamp = time.time()
            self.received_count += 1

    def latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            age = time.time() - self._stamp
            if age > self.max_age_sec:
                return None, age
            return self._frame, age


class ColoredCloudAccumulator:
    def __init__(self, voxel_size: float = 0.04, max_cells: int = 4_000_000):
        self.voxel_size = float(voxel_size)
        self.max_cells = int(max_cells)
        self._lock = threading.Lock()
        self._cells: dict = {}

    def add_frame(self, pts_world: np.ndarray, rgb: np.ndarray, colored_mask: np.ndarray):
        if pts_world.size == 0:
            return
        coords = np.floor(pts_world / self.voxel_size).astype(np.int64)
        with self._lock:
            for idx in range(len(pts_world)):
                key = (int(coords[idx, 0]), int(coords[idx, 1]), int(coords[idx, 2]))
                cell = self._cells.get(key)
                if cell is None:
                    self._cells[key] = [
                        pts_world[idx].astype(np.float64).copy(),
                        rgb[idx].astype(np.float64).copy(),
                        1,
                        1 if colored_mask[idx] else 0,
                    ]
                else:
                    cell[0] += pts_world[idx]
                    cell[1] += rgb[idx]
                    cell[2] += 1
                    if colored_mask[idx]:
                        cell[3] += 1

            if len(self._cells) > self.max_cells:
                drop_n = len(self._cells) - self.max_cells
                keys = list(self._cells.keys())[:drop_n]
                for k in keys:
                    self._cells.pop(k, None)

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            n = len(self._cells)
            if n == 0:
                return (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
            xyz = np.empty((n, 3), dtype=np.float32)
            rgb = np.empty((n, 3), dtype=np.float32)
            ratio = np.empty((n,), dtype=np.float32)
            for idx, (_, cell) in enumerate(self._cells.items()):
                cnt = max(cell[2], 1)
                xyz[idx] = cell[0] / cnt
                rgb[idx] = np.clip(cell[1] / cnt, 0.0, 1.0)
                ratio[idx] = cell[3] / cnt
            return xyz, rgb, ratio

    def __len__(self):
        with self._lock:
            return len(self._cells)


class KeyframeRecorder:
    def __init__(self, out_dir: Path, min_translation: float = 0.30, min_rotation_deg: float = 15.0,
                 min_interval_sec: float = 1.2):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.min_translation = float(min_translation)
        self.min_rotation_rad = math.radians(min_rotation_deg)
        self.min_interval_sec = float(min_interval_sec)
        self.last_pose: Optional[Pose] = None
        self.last_stamp = 0.0
        self.count = 0

    def maybe_save(self, frame_bgr: np.ndarray, pose: Pose, cam: CameraModel):
        now = time.time()
        if now - self.last_stamp < self.min_interval_sec:
            return
        if self.last_pose is not None:
            dt_pos = float(np.linalg.norm(pose.position - self.last_pose.position))
            yaw_diff = abs(((pose.rpy[2] - self.last_pose.rpy[2] + math.pi) % (2 * math.pi)) - math.pi)
            if dt_pos < self.min_translation and yaw_diff < self.min_rotation_rad:
                return

        self.count += 1
        idx = f"{self.count:04d}"
        img_path = self.out_dir / f"k_{idx}.jpg"
        meta_path = self.out_dir / f"k_{idx}.json"

        cv2.imwrite(str(img_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        meta = {
            "timestamp": now,
            "position": pose.position.tolist(),
            "rpy": pose.rpy.tolist(),
            "intrinsics": {
                "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                "width": cam.width, "height": cam.height,
            },
            "cam_in_lidar_xyz": cam.cam_offset_in_lidar.tolist(),
            "R_cam_from_lidar": cam.R_cam_lidar.tolist(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.last_pose = pose
        self.last_stamp = now


class AtomicCloudWriter:
    def __init__(self, target: Path):
        self.target = target
        self.target.parent.mkdir(parents=True, exist_ok=True)

    def write(self, xyz: np.ndarray, rgb: np.ndarray, colored_ratio: np.ndarray):
        tmp = self.target.with_suffix(self.target.suffix + ".tmp")
        np.savez_compressed(
            tmp,
            xyz=xyz.astype(np.float32),
            rgb=rgb.astype(np.float32),
            colored_ratio=colored_ratio.astype(np.float32),
        )
        os.replace(tmp, self.target)


# =========================================================
# PRODUCER
# =========================================================

class TexturedProducer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir = self.output_dir / "keyframes"

        self.camera = CameraModel(
            fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
            width=args.cam_width, height=args.cam_height,
            cam_offset_in_lidar=np.array([args.cam_dx, args.cam_dy, args.cam_dz], dtype=np.float64),
        )
        self.pose = PoseTracker(lidar_mount_offset_xyz=(0.0, 0.0, args.lidar_mount_z))
        self.video = VideoBuffer(max_age_sec=args.cam_max_age)
        self.cloud = ColoredCloudAccumulator(voxel_size=args.voxel_size, max_cells=args.max_cells)
        self.keyframes = KeyframeRecorder(
            self.keyframes_dir,
            min_translation=args.keyframe_min_translation,
            min_rotation_deg=args.keyframe_min_rotation_deg,
            min_interval_sec=args.keyframe_min_interval,
        )
        self.live_writer = AtomicCloudWriter(self.output_dir / "live_cloud.npz")

        self.range_filter = (args.min_range, args.max_range)
        self.z_filter = (args.z_min, args.z_max)

        self.running = True
        self.lidar_queue: "Queue[np.ndarray]" = Queue(maxsize=8)

        self.frame_count = 0
        self.colored_points_count = 0

    # ----- Callbacks invocados desde el thread de asyncio -----

    async def on_video_track(self, track: MediaStreamTrack):
        while self.running:
            try:
                frame = await track.recv()
            except Exception:
                return
            img = frame.to_ndarray(format="bgr24")
            if (img.shape[1] != self.camera.width) or (img.shape[0] != self.camera.height):
                img = cv2.resize(img, (self.camera.width, self.camera.height), interpolation=cv2.INTER_AREA)
            self.video.push(img)

    def on_pose_message(self, message: Any):
        try:
            if isinstance(message, dict):
                data = message.get("data", {})
                position = data.get("position", [0.0, 0.0, 0.0])
                rpy = data.get("rpy")
                if rpy is None:
                    rpy = data.get("imu_state", {}).get("rpy", [0.0, 0.0, 0.0])
                self.pose.update(position, rpy)
        except Exception:
            pass

    def on_lidar_message(self, message: Any):
        points = self._extract_points(message)
        if points is None or len(points) == 0:
            return
        try:
            self.lidar_queue.put_nowait(points)
        except Exception:
            try:
                _ = self.lidar_queue.get_nowait()
                self.lidar_queue.put_nowait(points)
            except Exception:
                pass

    def _extract_points(self, payload: Any) -> Optional[np.ndarray]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            topic = payload.get("topic")
            if topic == LIDAR_TOPIC:
                outer = payload.get("data", {})
                inner = outer.get("data", {})
                if "points" in inner:
                    return self._extract_points(inner["points"])

                origin = outer.get("origin")
                resolution = outer.get("resolution")
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

    # ----- Procesamiento (main thread) -----

    def process_lidar_frame(self, pts_lidar: np.ndarray):
        dist = np.linalg.norm(pts_lidar, axis=1)
        keep = (
            (dist >= self.range_filter[0]) & (dist <= self.range_filter[1])
            & (pts_lidar[:, 2] >= self.z_filter[0]) & (pts_lidar[:, 2] <= self.z_filter[1])
        )
        pts_lidar = pts_lidar[keep]
        if pts_lidar.size == 0:
            return

        frame_bgr, _age = self.video.latest()
        rgb = np.tile(NEUTRAL_COLOR, (len(pts_lidar), 1)).astype(np.float32)
        colored_mask = np.zeros(len(pts_lidar), dtype=bool)

        if frame_bgr is not None:
            uv, valid = self.camera.project_lidar_points(pts_lidar.astype(np.float64))
            if np.any(valid):
                u = uv[valid, 0]
                v = uv[valid, 1]
                sampled = frame_bgr[v, u]
                rgb[valid] = sampled[:, ::-1].astype(np.float32) / 255.0
                colored_mask[valid] = True
                self.colored_points_count += int(valid.sum())

        pose = self.pose.snapshot()
        T = pose.world_T_lidar(self.pose.lidar_mount)
        h = np.hstack([pts_lidar, np.ones((len(pts_lidar), 1), dtype=np.float32)])
        pts_world = (T @ h.T).T[:, :3]

        self.cloud.add_frame(pts_world, rgb, colored_mask)
        self.frame_count += 1

        if frame_bgr is not None and self.args.save_keyframes:
            self.keyframes.maybe_save(frame_bgr, pose, self.camera)

    def writer_loop(self):
        period = max(0.1, self.args.live_dump_interval)
        while self.running:
            time.sleep(period)
            xyz, rgb, ratio = self.cloud.snapshot()
            if len(xyz) > 0:
                self.live_writer.write(xyz, rgb, ratio)

    def stats_loop(self):
        while self.running:
            time.sleep(1.0)
            map_n = len(self.cloud)
            frame, age = self.video.latest()
            cam_age = f"{age*1000:.0f}ms" if frame is not None else "n/a"
            sys.stdout.write(
                f"\rframes={self.frame_count:6d} | "
                f"map_cells={map_n:7d} | "
                f"colored_pts={self.colored_points_count:9d} | "
                f"cam_age={cam_age} | "
                f"keyframes={self.keyframes.count:3d} | "
                f"pose={'REAL' if self.pose.have_real_pose else 'IDENTITY'}   "
            )
            sys.stdout.flush()

    def save_final(self):
        xyz, rgb, ratio = self.cloud.snapshot()
        if len(xyz) > 0:
            ply_path = self.output_dir / "final_cloud.ply"
            self._write_ply(ply_path, xyz, rgb)
            print(f"\n[OK] Nube final guardada en PLY Binario: {ply_path} ({len(xyz)} puntos)")

        traj = self.pose.trajectory
        if traj:
            traj_path = self.output_dir / "trajectory.json"
            with open(traj_path, "w") as f:
                json.dump(
                    [{"t": t, "position": [x, y, z], "rpy": [r, p, yw]} for (t, x, y, z, r, p, yw) in traj],
                    f, indent=2,
                )
            print(f"[OK] Trayectoria exportada en: {traj_path}")

    @staticmethod
    def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        n = len(xyz)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        buf = np.empty(n, dtype=dtype)
        buf["x"] = xyz[:, 0].astype(np.float32)
        buf["y"] = xyz[:, 1].astype(np.float32)
        buf["z"] = xyz[:, 2].astype(np.float32)
        buf["r"] = rgb8[:, 0]
        buf["g"] = rgb8[:, 1]
        buf["b"] = rgb8[:, 2]
        with open(path, "wb") as f:
            f.write(header)
            f.write(buf.tobytes())


# =========================================================
# SETUP DE SUSCRIPCIONES (API real de unitree_webrtc_connect)
# =========================================================

def setup_subscriptions(producer: TexturedProducer, conn: UnitreeWebRTCConnection):
    pubsub = conn.datachannel.pub_sub

    conn.datachannel.set_decoder(decoder_type=producer.args.lidar_decoder)
    pubsub.publish_without_callback(LIDAR_SWITCH_TOPIC, "on")
    print("[OK] Comando de activación enviado al LiDAR")

    pubsub.subscribe(LIDAR_TOPIC, producer.on_lidar_message)
    print(f"[OK] Suscripto a LiDAR en: {LIDAR_TOPIC}")

    pubsub.subscribe(POSE_TOPIC_PRIMARY, producer.on_pose_message)
    print(f"[OK] Suscripto a Pose en: {POSE_TOPIC_PRIMARY}")

    if POSE_TOPIC_FALLBACK != POSE_TOPIC_PRIMARY:
        pubsub.subscribe(POSE_TOPIC_FALLBACK, producer.on_pose_message)
        print(f"[OK] Suscripto a Pose fallback en: {POSE_TOPIC_FALLBACK}")


# =========================================================
# CONEXION (asyncio en thread aparte, igual a display_video_channel.py)
# =========================================================

def build_connection(args):
    if args.conn_mode == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
    if args.conn_mode == "sta":
        if args.ip:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
        if args.serial:
            return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial)
        raise ValueError("En modo STA debés proveer --ip o --serial.")
    if args.conn_mode == "remote":
        return UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )
    raise ValueError("Modo de conexión inválido.")


def run_asyncio_loop(loop: asyncio.AbstractEventLoop,
                     conn: UnitreeWebRTCConnection,
                     producer: TexturedProducer,
                     ready_event: threading.Event):
    """Mismo patron que display_video_channel.py: conectar + activar video,
    despues run_forever(). Aca tambien activamos LiDAR + suscripciones."""
    asyncio.set_event_loop(loop)

    async def setup():
        try:
            print("[INFO] Conectando vía WebRTC...")
            await conn.connect()
            print("[OK] Conexión WebRTC establecida.")

            # Video (igual que el script que funciona)
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(producer.on_video_track)
            print("[OK] Canal de video activado.")

            # Habilitar trafico de datos para que llegue el voxel_map.
            # Con timeout para no quedarnos colgados si el robot no responde.
            try:
                ok = await asyncio.wait_for(conn.datachannel.disableTrafficSaving(True), timeout=5.0)
                print(f"[OK] disableTrafficSaving -> {ok}")
            except asyncio.TimeoutError:
                print("[WARN] disableTrafficSaving timeout; sigo.")
            except Exception as e:
                print(f"[WARN] disableTrafficSaving falló: {e}; sigo.")

            # LiDAR + Pose
            setup_subscriptions(producer, conn)
        except Exception as e:
            print(f"[ERR] Falló el setup WebRTC: {type(e).__name__}: {e}")
            producer.running = False
        finally:
            ready_event.set()

    loop.run_until_complete(setup())
    if producer.running:
        loop.run_forever()


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline texturizado LiDAR + Cámara para Go2")
    p.add_argument("--conn-mode", choices=["ap", "sta", "remote"], default="ap")
    p.add_argument("--ip")
    p.add_argument("--serial")
    p.add_argument("--username")
    p.add_argument("--password")

    p.add_argument("--output-dir", default="scan_output")
    p.add_argument("--voxel-size", type=float, default=0.04)
    p.add_argument("--max-cells", type=int, default=4_000_000)
    p.add_argument("--live-dump-interval", type=float, default=0.5)
    p.add_argument("--lidar-decoder", choices=["libvoxel", "native"], default="libvoxel",
                   help="Decoder de LiDAR de unitree_webrtc_connect.")

    p.add_argument("--min-range", type=float, default=0.20)
    p.add_argument("--max-range", type=float, default=8.0)
    p.add_argument("--z-min", type=float, default=-0.40)
    p.add_argument("--z-max", type=float, default=2.50)

    p.add_argument("--fx", type=float, default=DEFAULT_INTRINSICS["fx"])
    p.add_argument("--fy", type=float, default=DEFAULT_INTRINSICS["fy"])
    p.add_argument("--cx", type=float, default=DEFAULT_INTRINSICS["cx"])
    p.add_argument("--cy", type=float, default=DEFAULT_INTRINSICS["cy"])
    p.add_argument("--cam-width", type=int, default=DEFAULT_INTRINSICS["width"])
    p.add_argument("--cam-height", type=int, default=DEFAULT_INTRINSICS["height"])
    p.add_argument("--cam-max-age", type=float, default=0.6)

    p.add_argument("--cam-dx", type=float, default=DEFAULT_CAM_OFFSET_XYZ[0])
    p.add_argument("--cam-dy", type=float, default=DEFAULT_CAM_OFFSET_XYZ[1])
    p.add_argument("--cam-dz", type=float, default=DEFAULT_CAM_OFFSET_XYZ[2])
    p.add_argument("--lidar-mount-z", type=float, default=0.30)

    p.add_argument("--save-keyframes", action="store_true", default=True)
    p.add_argument("--keyframe-min-translation", type=float, default=0.30)
    p.add_argument("--keyframe-min-rotation-deg", type=float, default=15.0)
    p.add_argument("--keyframe-min-interval", type=float, default=1.2)

    p.add_argument("--preview", action="store_true",
                   help="Mostrar la cámara en una ventana OpenCV (q para salir).")
    return p.parse_args()


def main():
    args = parse_args()
    producer = TexturedProducer(args)
    conn = build_connection(args)

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    asyncio_thread = threading.Thread(
        target=run_asyncio_loop,
        args=(loop, conn, producer, ready),
        daemon=True,
    )
    asyncio_thread.start()

    # Esperamos a que el setup termine (exitoso o no) antes de arrancar workers
    ready.wait(timeout=30.0)
    if not producer.running:
        print("[FATAL] No se pudo establecer la conexión. Abortando.")
        return

    # Manejador de Ctrl+C
    def _stop(*_):
        producer.running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Threads de fondo (escritura + stats)
    writer_thread = threading.Thread(target=producer.writer_loop, daemon=True)
    writer_thread.start()
    stats_thread = threading.Thread(target=producer.stats_loop, daemon=True)
    stats_thread.start()

    print(f"[INFO] Pipeline en ejecución. Salida en -> {producer.output_dir}")
    if args.preview:
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    # Main loop: procesa LiDAR + (opcional) preview de cámara
    try:
        while producer.running:
            # Procesar puntos LiDAR si hay
            try:
                pts = producer.lidar_queue.get(timeout=0.05)
                producer.process_lidar_frame(pts)
            except Empty:
                pass

            if args.preview:
                frame, _age = producer.video.latest()
                if frame is not None:
                    cv2.imshow("Video", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    producer.running = False
    finally:
        producer.running = False
        producer.save_final()

        if args.preview:
            cv2.destroyAllWindows()

        # Cerrar el loop de asyncio de forma limpia
        try:
            disc = getattr(conn, "disconnect", None)
            if callable(disc):
                fut = asyncio.run_coroutine_threadsafe(
                    _maybe_await(disc()), loop
                )
                try:
                    fut.result(timeout=3.0)
                except Exception:
                    pass
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=3.0)


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


if __name__ == "__main__":
    main()
