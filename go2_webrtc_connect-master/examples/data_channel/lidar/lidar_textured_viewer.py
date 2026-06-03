#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIEWER del pipeline de mapeo texturizado.

Corre en el venv que tiene Open3D (.venv_o3d).

Lee periodicamente el archivo live_cloud.npz que escribe el producer y muestra
el cloud coloreado en una ventana 3D interactiva. Atajos de teclado:

  R    : reconstruir malla Poisson con los puntos actuales y mostrarla
         (toggle entre cloud y mesh)
  T    : recargar trayectoria (trajectory.json) y mostrarla como linea
  C    : ciclar modo de color (camara / altura / mezcla)
  N    : alternar normales visibles
  S    : guardar snapshot PNG de la ventana en el cwd
  O    : alternar filtro de outliers en el cloud mostrado
  +/-  : aumentar/disminuir tamano de punto

Uso:
  python lidar_textured_viewer.py --input scan_output/live_cloud.npz
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


# =========================================================
# DATA LOADER
# =========================================================

def load_npz_cloud(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (xyz, rgb, colored_ratio)."""
    with np.load(path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        rgb = np.asarray(data["rgb"], dtype=np.float64)
        ratio = np.asarray(data["colored_ratio"], dtype=np.float64) if "colored_ratio" in data.files \
            else np.ones((len(xyz),), dtype=np.float64)
    return xyz, rgb, ratio


def height_colormap(xyz: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 1e-6:
        return np.tile([0.2, 0.8, 1.0], (len(xyz), 1))
    zn = (z - zmin) / (zmax - zmin)
    colors = np.zeros((len(xyz), 3), dtype=np.float64)
    colors[:, 0] = np.clip(1.8 * zn, 0, 1)
    colors[:, 1] = np.clip(1.6 * (1 - np.abs(zn - 0.5) * 2), 0, 1)
    colors[:, 2] = np.clip(1.8 * (1 - zn), 0, 1)
    return colors


def blend_colors(rgb_cam: np.ndarray, ratio: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Donde no tenemos color de camara, usamos colormap por altura."""
    h_col = height_colormap(xyz)
    has_cam = (ratio > 0.05)[:, None]
    return np.where(has_cam, rgb_cam, h_col)


# =========================================================
# RECONSTRUCTION HELPERS
# =========================================================

def reconstruct_mesh(pcd: o3d.geometry.PointCloud,
                     depth: int = 9,
                     density_quantile: float = 0.04) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction con densidad para limpiar artefactos."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(50)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False
    )
    densities = np.asarray(densities)
    if len(densities) > 0:
        threshold = np.quantile(densities, density_quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    return mesh


# =========================================================
# VIEWER
# =========================================================

class LiveViewer:
    COLOR_MODES = ("camera", "height", "blend")

    def __init__(self, npz_path: Path, traj_path: Optional[Path],
                 poll_interval: float, point_size: float):
        self.npz_path = npz_path
        self.traj_path = traj_path
        self.poll_interval = poll_interval
        self.point_size = point_size

        self.last_mtime: Optional[float] = None
        self.last_cloud_n = 0

        self.pcd = o3d.geometry.PointCloud()
        self.traj_line = o3d.geometry.LineSet()

        self.show_mesh = False
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None

        self.color_mode_idx = 2  # blend
        self.show_normals = False
        self.show_trajectory = True
        self.outlier_removal = False

        self.last_xyz: Optional[np.ndarray] = None
        self.last_rgb_cam: Optional[np.ndarray] = None
        self.last_ratio: Optional[np.ndarray] = None

        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self._stats_t = 0.0

    # ---------------------------------------------------------

    def _apply_cloud_colors(self) -> np.ndarray:
        if self.last_xyz is None:
            return np.zeros((0, 3))
        mode = self.COLOR_MODES[self.color_mode_idx]
        if mode == "camera":
            return self.last_rgb_cam
        if mode == "height":
            return height_colormap(self.last_xyz)
        return blend_colors(self.last_rgb_cam, self.last_ratio, self.last_xyz)

    def _maybe_outlier_clean(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if not self.outlier_removal or len(pcd.points) < 200:
            return pcd
        cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.0)
        return cleaned

    def _refresh_cloud(self, vis):
        if not self.npz_path.exists():
            return False
        try:
            mtime = os.path.getmtime(self.npz_path)
        except OSError:
            return False
        if self.last_mtime is not None and mtime == self.last_mtime:
            return False
        try:
            xyz, rgb, ratio = load_npz_cloud(self.npz_path)
        except Exception:
            return False
        if len(xyz) == 0:
            return False

        self.last_mtime = mtime
        self.last_xyz = xyz
        self.last_rgb_cam = rgb
        self.last_ratio = ratio

        colors = self._apply_cloud_colors()
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if self.show_normals or self.show_mesh:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
            )

        if self.outlier_removal:
            cleaned = self._maybe_outlier_clean(self.pcd)
            self.pcd.points = cleaned.points
            self.pcd.colors = cleaned.colors
            if cleaned.has_normals():
                self.pcd.normals = cleaned.normals

        vis.update_geometry(self.pcd)
        first = (self.last_cloud_n == 0)
        self.last_cloud_n = len(xyz)
        if first:
            vis.reset_view_point(True)
        return True

    def _refresh_trajectory(self, vis):
        if self.traj_path is None or not self.traj_path.exists():
            return
        try:
            with open(self.traj_path, "r") as f:
                data = json.load(f)
        except Exception:
            return
        if not data:
            return
        pts = np.array([d["position"] for d in data if "position" in d], dtype=np.float64)
        if len(pts) < 2:
            return
        lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
        self.traj_line.points = o3d.utility.Vector3dVector(pts)
        self.traj_line.lines = o3d.utility.Vector2iVector(lines)
        self.traj_line.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.5, 0.0], (len(lines), 1))
        )
        vis.update_geometry(self.traj_line)

    def _recolor_only(self, vis):
        if self.last_xyz is None:
            return
        colors = self._apply_cloud_colors()
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(self.pcd)

    # ---- key callbacks ---------------------------------------

    def _toggle_mesh(self, vis):
        if not self.show_mesh:
            if self.last_xyz is None or len(self.last_xyz) < 500:
                print("\n[mesh] no hay puntos suficientes para reconstruir todavia.")
                return False
            print("\n[mesh] reconstruyendo Poisson... (puede tardar)")
            t0 = time.time()
            try:
                mesh = reconstruct_mesh(self.pcd, depth=9, density_quantile=0.04)
                # Transfer colores por nearest neighbor desde el cloud.
                kdt = o3d.geometry.KDTreeFlann(self.pcd)
                v = np.asarray(mesh.vertices)
                src_colors = np.asarray(self.pcd.colors)
                vc = np.zeros((len(v), 3), dtype=np.float64)
                for i in range(len(v)):
                    _, idx, _ = kdt.search_knn_vector_3d(v[i], 1)
                    if idx:
                        vc[i] = src_colors[idx[0]]
                mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
                self.mesh = mesh
                vis.add_geometry(mesh, reset_bounding_box=False)
                vis.remove_geometry(self.pcd, reset_bounding_box=False)
                self.show_mesh = True
                print(f"[mesh] listo en {time.time()-t0:.1f}s "
                      f"({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos)")
            except Exception as e:
                print(f"[mesh] error: {e}")
        else:
            if self.mesh is not None:
                vis.remove_geometry(self.mesh, reset_bounding_box=False)
                self.mesh = None
            vis.add_geometry(self.pcd, reset_bounding_box=False)
            self.show_mesh = False
            print("\n[mesh] volviendo al cloud")
        return False

    def _cycle_color(self, vis):
        self.color_mode_idx = (self.color_mode_idx + 1) % len(self.COLOR_MODES)
        mode = self.COLOR_MODES[self.color_mode_idx]
        print(f"\n[color] modo = {mode}")
        self._recolor_only(vis)
        return False

    def _toggle_normals(self, vis):
        self.show_normals = not self.show_normals
        opt = vis.get_render_option()
        opt.point_show_normal = self.show_normals
        if self.show_normals and not self.pcd.has_normals() and self.last_xyz is not None:
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
            )
            vis.update_geometry(self.pcd)
        print(f"\n[normals] {'on' if self.show_normals else 'off'}")
        return False

    def _toggle_outliers(self, vis):
        self.outlier_removal = not self.outlier_removal
        print(f"\n[outliers] filtro {'ON' if self.outlier_removal else 'OFF'} (se aplica al proximo refresh)")
        self.last_mtime = None  # force refresh
        self._refresh_cloud(vis)
        return False

    def _toggle_trajectory(self, vis):
        self.show_trajectory = not self.show_trajectory
        if self.show_trajectory:
            vis.add_geometry(self.traj_line, reset_bounding_box=False)
            self._refresh_trajectory(vis)
        else:
            vis.remove_geometry(self.traj_line, reset_bounding_box=False)
        print(f"\n[trajectory] {'ON' if self.show_trajectory else 'OFF'}")
        return False

    def _snapshot(self, vis):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.png"
        vis.capture_screen_image(path, do_render=True)
        print(f"\n[snapshot] guardado en {path}")
        return False

    def _bigger_points(self, vis):
        opt = vis.get_render_option()
        opt.point_size = min(opt.point_size + 0.5, 12.0)
        return False

    def _smaller_points(self, vis):
        opt = vis.get_render_option()
        opt.point_size = max(opt.point_size - 0.5, 1.0)
        return False

    # ---- animation callback ----------------------------------

    def _animation_tick(self, vis):
        # poll filesystem para nuevos datos
        if self.show_mesh:
            return False
        changed = self._refresh_cloud(vis)
        if self.show_trajectory:
            self._refresh_trajectory(vis)
        # stats cada ~2s
        now = time.time()
        if now - self._stats_t > 2.0 and self.last_xyz is not None:
            self._stats_t = now
            colored = int((self.last_ratio > 0).sum()) if self.last_ratio is not None else 0
            mins = self.last_xyz.min(axis=0)
            maxs = self.last_xyz.max(axis=0)
            print(
                f"\r[viewer] pts={len(self.last_xyz):7d} "
                f"color_cam={colored:7d} | "
                f"X[{mins[0]: .2f},{maxs[0]: .2f}] "
                f"Y[{mins[1]: .2f},{maxs[1]: .2f}] "
                f"Z[{mins[2]: .2f},{maxs[2]: .2f}]      ",
                end="", flush=True,
            )
        return changed

    # ---- run ---------------------------------------------------

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis = vis
        vis.create_window(window_name="Go2 Textured Map (viewer)", width=1400, height=900)
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.asarray([0.04, 0.05, 0.07])
        opt.light_on = True

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(axis)
        vis.add_geometry(self.pcd)
        if self.show_trajectory:
            vis.add_geometry(self.traj_line)

        # carga inicial si el archivo ya existe
        if self.npz_path.exists():
            self._refresh_cloud(vis)
            self._refresh_trajectory(vis)

        # bindings
        vis.register_key_callback(ord("R"), self._toggle_mesh)
        vis.register_key_callback(ord("T"), self._toggle_trajectory)
        vis.register_key_callback(ord("C"), self._cycle_color)
        vis.register_key_callback(ord("N"), self._toggle_normals)
        vis.register_key_callback(ord("O"), self._toggle_outliers)
        vis.register_key_callback(ord("S"), self._snapshot)
        vis.register_key_callback(ord("+"), self._bigger_points)
        vis.register_key_callback(ord("="), self._bigger_points)  # tecla = en layouts ES
        vis.register_key_callback(ord("-"), self._smaller_points)

        vis.register_animation_callback(self._animation_tick)

        print("[viewer] R=mesh  T=traj  C=color  N=normals  O=outliers  S=snapshot  +/- size  Q=quit")
        vis.run()
        vis.destroy_window()


# =========================================================
# MAIN
# =========================================================

def main():
    p = argparse.ArgumentParser(description="Viewer en vivo del cloud texturizado")
    p.add_argument("--input", required=True, help="Ruta a live_cloud.npz")
    p.add_argument("--trajectory", default=None,
                   help="Ruta a trajectory.json (default: junto al input)")
    p.add_argument("--poll", type=float, default=0.4)
    p.add_argument("--point-size", type=float, default=3.0)
    args = p.parse_args()

    npz_path = Path(args.input).resolve()
    if args.trajectory:
        traj_path = Path(args.trajectory).resolve()
    else:
        traj_path = npz_path.parent / "trajectory.json"

    print(f"[viewer] input    : {npz_path}")
    print(f"[viewer] trajectory: {traj_path} {'(existe)' if traj_path.exists() else '(no existe aun)'}")

    viewer = LiveViewer(
        npz_path=npz_path,
        traj_path=traj_path,
        poll_interval=args.poll,
        point_size=args.point_size,
    )
    viewer.run()


if __name__ == "__main__":
    main()
