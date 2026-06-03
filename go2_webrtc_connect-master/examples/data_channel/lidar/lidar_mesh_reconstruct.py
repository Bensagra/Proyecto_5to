#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECONSTRUCCION OFFLINE de la malla texturizada.

Corre en el venv que tiene Open3D (.venv_o3d).

Toma el cloud final guardado por el producer (PLY o NPZ), lo limpia, estima
normales con orientacion consistente, corre Poisson surface reconstruction y
exporta una malla con colores por vertice (PLY y OBJ).

Opcionalmente, re-proyecta los keyframes de camara sobre la malla para refinar
el color de cada vertice (multi-view color averaging). Eso da una textura
mas estable que un solo frame por punto.

Uso minimo:
  python lidar_mesh_reconstruct.py --input scan_output/final_cloud.ply

Refinando con keyframes:
  python lidar_mesh_reconstruct.py \
      --input scan_output/final_cloud.ply \
      --keyframes scan_output/keyframes \
      --output-prefix scan_output/mesh

Tuning relevante:
  --depth 10                 mas detalle (mas RAM, mas tiempo). Default 9.
  --density-quantile 0.05    sube esto si quedan superficies espureas.
  --voxel-size 0.03          downsample previo para acelerar Poisson.
  --normal-radius 0.20       radio de busqueda al estimar normales.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


# =========================================================
# LOADERS
# =========================================================

def load_input_cloud(path: Path) -> o3d.geometry.PointCloud:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            rgb = np.asarray(data["rgb"], dtype=np.float64) if "rgb" in data.files \
                else np.ones_like(xyz) * 0.6
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
        return pcd
    return o3d.io.read_point_cloud(str(path))


def load_keyframes(folder: Path) -> List[dict]:
    """Devuelve lista de {image_path, pose, intrinsics, R_cam_lidar, cam_in_lidar}."""
    if not folder.exists():
        return []
    metas = sorted(glob.glob(str(folder / "k_*.json")))
    out = []
    for m in metas:
        try:
            with open(m, "r") as f:
                meta = json.load(f)
            img_path = Path(m).with_suffix(".jpg")
            if not img_path.exists():
                continue
            out.append({
                "image_path": img_path,
                "position": np.array(meta["position"], dtype=np.float64),
                "rpy": np.array(meta["rpy"], dtype=np.float64),
                "intrinsics": meta["intrinsics"],
                "R_cam_lidar": np.array(meta["R_cam_from_lidar"], dtype=np.float64),
                "cam_in_lidar": np.array(meta["cam_in_lidar_xyz"], dtype=np.float64),
            })
        except Exception as e:
            print(f"[warn] keyframe inval {m}: {e}")
    return out


# =========================================================
# CLEANING
# =========================================================

def clean_cloud(pcd: o3d.geometry.PointCloud,
                voxel_size: float,
                nb_neighbors: int,
                std_ratio: float,
                radius: float,
                min_neighbors: int) -> o3d.geometry.PointCloud:
    n0 = len(pcd.points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if radius > 0 and min_neighbors > 0:
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    print(f"[clean] {n0} -> {len(pcd.points)} puntos")
    return pcd


# =========================================================
# NORMALS + POISSON
# =========================================================

def compute_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int,
                    orient_k: int) -> None:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(orient_k)


def run_poisson(pcd: o3d.geometry.PointCloud, depth: int, density_quantile: float
                ) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False
    )
    densities = np.asarray(densities)
    if len(densities) > 0:
        thr = np.quantile(densities, density_quantile)
        mask = densities < thr
        mesh.remove_vertices_by_mask(mask)
    mesh.compute_vertex_normals()
    return mesh


# =========================================================
# COLOR TRANSFER
# =========================================================

def transfer_colors_nearest(mesh: o3d.geometry.TriangleMesh,
                            source: o3d.geometry.PointCloud) -> None:
    if len(source.points) == 0:
        return
    kdt = o3d.geometry.KDTreeFlann(source)
    v = np.asarray(mesh.vertices)
    src_colors = np.asarray(source.colors)
    vc = np.zeros((len(v), 3), dtype=np.float64)
    for i in range(len(v)):
        _, idx, _ = kdt.search_knn_vector_3d(v[i], 1)
        if idx:
            vc[i] = src_colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vc)


# =========================================================
# MULTI-VIEW COLOR REFINEMENT (keyframes)
# =========================================================

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    import math
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def refine_colors_with_keyframes(mesh: o3d.geometry.TriangleMesh,
                                 keyframes: List[dict],
                                 lidar_mount_z: float = 0.30,
                                 weight_blend: float = 0.55) -> None:
    """
    Blendea el color por vertice mezclando los colores re-proyectados desde
    cada keyframe. Pondera por:
      - depth (cerca > lejos)
      - posicion central en la imagen (centro > borde)
      - normal alineada a la camara (apunta a > apunta lejos)

    El parametro weight_blend controla cuanto pesa el color refinado vs el
    color original que ya traia el vertice (de la fusion online).
    """
    try:
        import cv2
    except ImportError:
        print("[refine] opencv no esta instalado en este venv, salto el refine de keyframes.")
        return

    if not keyframes:
        return

    v = np.asarray(mesh.vertices, dtype=np.float64)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    base_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if base_colors.shape != v.shape:
        base_colors = np.ones_like(v) * 0.55

    accum = np.zeros_like(v)
    weights = np.zeros((len(v),), dtype=np.float64)

    for k in keyframes:
        img = cv2.imread(str(k["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        fx = k["intrinsics"]["fx"]; fy = k["intrinsics"]["fy"]
        cx = k["intrinsics"]["cx"]; cy = k["intrinsics"]["cy"]
        K_W = k["intrinsics"]["width"]; K_H = k["intrinsics"]["height"]
        if (W != K_W) or (H != K_H):
            img = cv2.resize(img, (K_W, K_H), interpolation=cv2.INTER_AREA)
            H, W = K_H, K_W

        # world -> lidar: invertir la pose
        R_wl = rpy_to_R(*k["rpy"])
        t_wl = k["position"] + np.array([0.0, 0.0, lidar_mount_z], dtype=np.float64)
        # world -> lidar: P_lidar = R_wl.T @ (P_w - t_wl)
        v_lidar = (R_wl.T @ (v - t_wl).T).T

        # lidar -> camara
        R_cl = k["R_cam_lidar"]
        t_cam_in_lidar = k["cam_in_lidar"]
        p_cam = (R_cl @ (v_lidar - t_cam_in_lidar).T).T
        z = p_cam[:, 2]
        valid_depth = (z > 0.15) & (z < 10.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * (p_cam[:, 0] / z) + cx
            v_img = fy * (p_cam[:, 1] / z) + cy
        ui = np.round(u).astype(np.int32)
        vi = np.round(v_img).astype(np.int32)
        in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        valid = valid_depth & in_img
        if not np.any(valid):
            continue

        sampled_bgr = img[vi[valid], ui[valid]]
        sampled_rgb = sampled_bgr[:, ::-1].astype(np.float64) / 255.0

        # peso por depth, por centro de imagen y por alineacion de normal
        depth_w = 1.0 / np.maximum(z[valid], 0.5)
        center_w = np.exp(-((u[valid] - cx) ** 2 + (v_img[valid] - cy) ** 2) / (0.6 * (W ** 2)))

        # normal del vertice en frame camara
        n_cam = (R_cl @ R_wl.T @ vn.T).T[valid]
        # camara mira a +z; vertices con normal apuntando a -z (cara visible) son los buenos
        align = np.clip(-n_cam[:, 2], 0.05, 1.0)

        w = depth_w * center_w * align
        accum[valid] += sampled_rgb * w[:, None]
        weights[valid] += w

    visited = weights > 1e-6
    if not np.any(visited):
        print("[refine] ningun vertice fue visto por keyframes. Dejo colores como estaban.")
        return

    refined = base_colors.copy()
    refined[visited] = accum[visited] / weights[visited, None]
    blended = base_colors.copy()
    blended[visited] = (
        weight_blend * refined[visited]
        + (1.0 - weight_blend) * base_colors[visited]
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(blended, 0, 1))
    print(f"[refine] color refinado con {len(keyframes)} keyframes, "
          f"{int(visited.sum())}/{len(v)} vertices actualizados")


# =========================================================
# OUTPUT
# =========================================================

def save_mesh(mesh: o3d.geometry.TriangleMesh, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ply = prefix.with_suffix(".ply")
    obj = prefix.with_suffix(".obj")
    o3d.io.write_triangle_mesh(str(ply), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(obj), mesh, write_vertex_colors=True)
    print(f"[out] {ply}")
    print(f"[out] {obj}")


# =========================================================
# PIPELINE
# =========================================================

def pipeline(args) -> None:
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}")
    print(f"[in] {in_path}")

    pcd = load_input_cloud(in_path)
    if len(pcd.points) == 0:
        print("[err] cloud vacio")
        return
    print(f"[in] {len(pcd.points)} puntos cargados")

    t0 = time.time()
    pcd = clean_cloud(
        pcd,
        voxel_size=args.voxel_size,
        nb_neighbors=args.outlier_neighbors,
        std_ratio=args.outlier_std,
        radius=args.radius_outlier_radius,
        min_neighbors=args.radius_outlier_min_neighbors,
    )

    print(f"[normals] estimando (radius={args.normal_radius})...")
    compute_normals(
        pcd,
        radius=args.normal_radius,
        max_nn=args.normal_max_nn,
        orient_k=args.normal_orient_k,
    )

    print(f"[poisson] depth={args.depth} density_q={args.density_quantile}")
    mesh = run_poisson(pcd, depth=args.depth, density_quantile=args.density_quantile)
    print(f"[poisson] {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos")

    transfer_colors_nearest(mesh, pcd)

    if args.keyframes:
        kf_dir = Path(args.keyframes).resolve()
        kfs = load_keyframes(kf_dir)
        print(f"[refine] cargados {len(kfs)} keyframes de {kf_dir}")
        if kfs:
            refine_colors_with_keyframes(
                mesh, kfs,
                lidar_mount_z=args.lidar_mount_z,
                weight_blend=args.keyframe_blend,
            )

    prefix = Path(args.output_prefix).resolve()
    save_mesh(mesh, prefix)

    print(f"[done] total {time.time()-t0:.1f}s")
    if args.preview:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Mesh reconstruida",
            mesh_show_back_face=True,
        )


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Reconstruccion offline de malla texturizada")
    p.add_argument("--input", required=True, help="Cloud PLY o NPZ")
    p.add_argument("--output-prefix", default="scan_output/mesh",
                   help="Prefijo de salida (.ply y .obj se anaden)")
    p.add_argument("--keyframes", default=None,
                   help="Carpeta con keyframes k_*.jpg + k_*.json para refinar color")

    p.add_argument("--voxel-size", type=float, default=0.03)
    p.add_argument("--outlier-neighbors", type=int, default=20)
    p.add_argument("--outlier-std", type=float, default=2.0)
    p.add_argument("--radius-outlier-radius", type=float, default=0.0,
                   help="0 desactiva el radius outlier filter")
    p.add_argument("--radius-outlier-min-neighbors", type=int, default=8)

    p.add_argument("--normal-radius", type=float, default=0.20)
    p.add_argument("--normal-max-nn", type=int, default=30)
    p.add_argument("--normal-orient-k", type=int, default=50)

    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--density-quantile", type=float, default=0.04)

    p.add_argument("--lidar-mount-z", type=float, default=0.30)
    p.add_argument("--keyframe-blend", type=float, default=0.55,
                   help="Peso del color refinado vs el original (0..1)")

    p.add_argument("--preview", action="store_true", default=True)
    p.add_argument("--no-preview", dest="preview", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline(args)


if __name__ == "__main__":
    main()
