import argparse
import os
import time
import numpy as np
import open3d as o3d


def make_colors_by_height(pts):
    z = pts[:, 2]
    zmin, zmax = z.min(), z.max()

    if zmax - zmin < 1e-9:
        return np.tile(np.array([[0.2, 0.8, 1.0]], dtype=np.float64), (len(pts), 1))

    zn = (z - zmin) / (zmax - zmin)
    colors = np.zeros((len(pts), 3), dtype=np.float64)
    colors[:, 0] = np.clip(1.8 * zn, 0, 1)
    colors[:, 1] = np.clip(1.6 * (1 - np.abs(zn - 0.5) * 2), 0, 1)
    colors[:, 2] = np.clip(1.8 * (1 - zn), 0, 1)
    return colors


def load_points(path):
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Formato inválido: {pts.shape}")
    return pts[:, :3].astype(np.float64)


class LiveMapVisualizer:
    def __init__(self, npy_path, poll_interval=0.15, point_size=4.0):
        self.npy_path = os.path.abspath(npy_path)
        self.poll_interval = poll_interval
        self.point_size = point_size
        self.last_mtime = None
        self.first_update = True
        self.pcd = o3d.geometry.PointCloud()
        
        print(f"[INFO] Monitoreando: {self.npy_path}")
        
        if not os.path.exists(self.npy_path):
            raise FileNotFoundError(f"Archivo no encontrado: {self.npy_path}")

    def update_geometry(self, vis):
        """Callback llamado en cada frame de animación"""
        if not os.path.exists(self.npy_path):
            return False
        
        try:
            mtime = os.path.getmtime(self.npy_path)
            
            # Solo actualizar si el archivo cambió
            if self.last_mtime is not None and mtime == self.last_mtime:
                return False
            
            # Cargar puntos nuevos
            pts = load_points(self.npy_path)
            
            if len(pts) == 0:
                return False
            
            # Colorear por altura
            colors = make_colors_by_height(pts)
            
            # Actualizar geometría
            self.pcd.points = o3d.utility.Vector3dVector(pts)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Notificar cambios
            vis.update_geometry(self.pcd)
            
            # Solo resetear bounding box en la primera actualización
            if self.first_update:
                vis.reset_view_point(True)
                self.first_update = False
            
            self.last_mtime = mtime
            print(f"[INFO] Actualizado - Puntos: {len(pts)}")
            
        except Exception as e:
            print(f"[WARN] Error actualizando nube: {e}")
        
        return False

    def run(self):
        """Iniciar visualización con actualizaciones automáticas"""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Go2 Live Map", width=1400, height=900)
        
        # Configurar opciones de render
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.asarray([0.02, 0.02, 0.02])
        
        # Cargar datos iniciales
        try:
            pts = load_points(self.npy_path)
            if len(pts) > 0:
                colors = make_colors_by_height(pts)
                self.pcd.points = o3d.utility.Vector3dVector(pts)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self.last_mtime = os.path.getmtime(self.npy_path)
        except Exception as e:
            print(f"[WARN] No se pudo cargar datos iniciales: {e}")
        
        # Agregar geometría
        vis.add_geometry(self.pcd)
        
        # Ajustar vista inicial
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        
        # Registrar callback de actualización
        vis.register_animation_callback(self.update_geometry)
        
        print("[INFO] Visualizador iniciado. Presiona Q o cierra la ventana para salir.")
        
        # Loop principal
        vis.run()
        vis.destroy_window()


def main():
    p = argparse.ArgumentParser(description="Visualizador en tiempo real de nube de puntos Go2")
    p.add_argument("--live-npy", required=True, help="Ruta al archivo .npy que se actualiza")
    p.add_argument("--poll", type=float, default=0.15, help="Intervalo de polling (segundos)")
    p.add_argument("--point-size", type=float, default=4.0, help="Tamaño de los puntos")
    args = p.parse_args()

    visualizer = LiveMapVisualizer(
        npy_path=args.live_npy,
        poll_interval=args.poll,
        point_size=args.point_size
    )
    
    visualizer.run()


if __name__ == "__main__":
    main()