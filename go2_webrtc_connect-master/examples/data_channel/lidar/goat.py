import asyncio
import math
import time
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import open3d as o3d

from go2_webrtc_driver.webrtc_driver import (
    Go2WebRTCConnection,
    WebRTCConnectionMethod,
)
from go2_webrtc_driver.constants import RTC_TOPIC


# =========================================================
# CONFIG
# =========================================================

ROBOT_IP = "192.168.8.181"


@dataclass
class MapperConfig:
    voxel_size: float = 0.05
    min_range_m: float = 0.15
    max_range_m: float = 8.0
    z_min_m: float = -0.30
    z_max_m: float = 2.50
    lidar_offset_xyz: tuple = (0.0, 0.0, 0.30)
    save_path: str = "mapa_3d_go2.ply"
    save_every_n_frames: int = 25


# =========================================================
# UTILS
# =========================================================

def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float64)

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return rz @ ry @ rx


def build_transform(position_xyz, rpy_xyz, lidar_offset_xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    px, py, pz = position_xyz
    roll, pitch, yaw = rpy_xyz

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rotation_matrix(roll, pitch, yaw)
    T[:3, 3] = [px, py, pz]

    T_lidar = np.eye(4, dtype=np.float64)
    T_lidar[:3, 3] = list(lidar_offset_xyz)

    return T @ T_lidar


def transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz

    pts_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)])
    pts_world = (T @ pts_h.T).T
    return pts_world[:, :3]


# =========================================================
# MAPPER
# =========================================================

class LidarMapper3D:
    def __init__(self, config: Optional[MapperConfig] = None):
        self.config = config or MapperConfig()

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rpy = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.global_points = []
        self.frame_count = 0
        self.last_save_time = time.time()

    def update_pose(self, position_xyz, rpy_xyz):
        self.position = np.array(position_xyz[:3], dtype=np.float64)
        self.rpy = np.array(rpy_xyz[:3], dtype=np.float64)

    def filter_points(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float64)

        dist = np.linalg.norm(pts[:, :3], axis=1)

        keep = (
            (dist >= self.config.min_range_m) &
            (dist <= self.config.max_range_m) &
            (pts[:, 2] >= self.config.z_min_m) &
            (pts[:, 2] <= self.config.z_max_m)
        )

        pts = pts[keep]
        return pts

    def add_lidar_frame(self, points_xyz: np.ndarray):
        local_pts = self.filter_points(points_xyz)
        if local_pts.size == 0:
            return

        T = build_transform(
            self.position,
            self.rpy,
            self.config.lidar_offset_xyz,
        )

        world_pts = transform_points(local_pts, T)
        self.global_points.append(world_pts)
        self.frame_count += 1

        if self.frame_count % self.config.save_every_n_frames == 0:
            print(f"[MAP] frames={self.frame_count}")

    def build_open3d_cloud(self):
        if not self.global_points:
            return o3d.geometry.PointCloud()

        all_points = np.vstack(self.global_points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)

        pcd = pcd.voxel_down_sample(voxel_size=self.config.voxel_size)
        return pcd

    def save(self, path: Optional[str] = None):
        path = path or self.config.save_path
        pcd = self.build_open3d_cloud()

        if len(pcd.points) == 0:
            print("[MAP] No hay puntos para guardar.")
            return False

        ok = o3d.io.write_point_cloud(path, pcd)
        if ok:
            print(f"[MAP] Guardado en {path}")
        else:
            print("[MAP] Error guardando mapa.")
        return ok

    def preview(self):
        pcd = self.build_open3d_cloud()
        if len(pcd.points) == 0:
            print("[MAP] No hay puntos para mostrar.")
            return
        o3d.visualization.draw_geometries([pcd])


# =========================================================
# CALLBACKS
# =========================================================

mapper = LidarMapper3D()


def safe_extract_data(payload: Any):
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], dict):
            return payload["data"]
        return payload
    return {}


def sport_state_callback(msg):
    data = safe_extract_data(msg)

    position = data.get("position", [0.0, 0.0, 0.0])

    rpy = data.get("rpy")
    if rpy is None:
        imu_state = data.get("imu_state", {})
        rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

    if len(position) >= 3 and len(rpy) >= 3:
        mapper.update_pose(position[:3], rpy[:3])


def lidar_callback(points_xyz):
    """
    Este callback espera que points_xyz ya sea un np.ndarray Nx3.
    """
    try:
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3:
            mapper.add_lidar_frame(pts)
    except Exception as e:
        print(f"[LIDAR] Error procesando frame: {e}")


# =========================================================
# SUBSCRIPTIONS
# =========================================================

async def subscribe_sport_state(conn):
    pubsub = conn.datachannel.pub_sub
    topic = RTC_TOPIC["LF_SPORT_MOD_STATE"]

    try:
        result = pubsub.subscribe(topic, sport_state_callback)
        if asyncio.iscoroutine(result):
            await result
        print("[STATE] Suscripto a LF_SPORT_MOD_STATE")
        return
    except Exception:
        pass

    try:
        result = pubsub.subscribe(topic=topic, callback=sport_state_callback)
        if asyncio.iscoroutine(result):
            await result
        print("[STATE] Suscripto a LF_SPORT_MOD_STATE")
        return
    except Exception as e:
        raise RuntimeError(f"No pude subscribirme a LF_SPORT_MOD_STATE: {e}")


async def subscribe_lidar(conn):
    """
    Acá hay diferencias entre versiones/forks.
    La idea es suscribirse al stream LiDAR oficial.
    Si tu ejemplo oficial usa otro método, reemplazá solo esta parte.
    """
    lidar = getattr(conn, "lidar", None)
    if lidar is None:
        raise RuntimeError("La conexión no expone módulo lidar.")

    # Caso A: método estilo callback
    if hasattr(lidar, "add_data_callback"):
        lidar.add_data_callback(lidar_callback)

    elif hasattr(lidar, "set_callback"):
        lidar.set_callback(lidar_callback)

    elif hasattr(lidar, "add_callback"):
        lidar.add_callback(lidar_callback)

    else:
        raise RuntimeError(
            "No encontré método conocido para registrar callback LiDAR. "
            "Abrí tu lidar_stream.py oficial y fijate cómo conectan el decoder."
        )

    # Activación del stream
    if hasattr(lidar, "switch_lidar"):

        result = lidar.switch_lidar(True)
        if asyncio.iscoroutine(result):
            await result

    elif hasattr(lidar, "enable"):

        result = lidar.enable()
        if asyncio.iscoroutine(result):
            await result

    elif hasattr(lidar, "switchLidar"):

        result = lidar.switchLidar(True)
        if asyncio.iscoroutine(result):
            await result

    print("[LIDAR] Stream LiDAR activado")


# =========================================================
# MAIN
# =========================================================

async def main():
    conn = Go2WebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP,
    )

    await conn.connect()
    print("[INFO] Conectado al robot")

    await subscribe_sport_state(conn)
    await subscribe_lidar(conn)

    print("[INFO] Empezá a mover el robot despacio por el ambiente.")
    print("[INFO] Hacé recorridos lentos, con giros suaves.")
    print("[INFO] Ctrl+C para guardar el mapa.")

    try:
        while True:
            await asyncio.sleep(1.0)
            print(f"[MAP] Frames acumulados: {mapper.frame_count}")
    except KeyboardInterrupt:
        print("\n[INFO] Guardando mapa...")
        mapper.save()
        mapper.preview()
    finally:
        try:
            await conn.disconnect()
        except Exception:
            pass
        print("[INFO] Desconectado")


if __name__ == "__main__":
    asyncio.run(main())