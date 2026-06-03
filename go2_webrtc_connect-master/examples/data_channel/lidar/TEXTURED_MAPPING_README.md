# Mapeo 3D texturizado con LiDAR + camara (Go2)

Pipeline de tres pasos que produce una reconstruccion 3D coloreada del ambiente
usando el LiDAR L1 y la camara frontal del Go2.

```
+-------------------------+         +--------------------------+
|  lidar_textured_        |  npz +  |  lidar_textured_         |
|  producer.py            | ----->  |  viewer.py               |
|  (.venv: webrtc+cv2)    |  PLY    |  (.venv_o3d: open3d)     |
+-----------+-------------+         +--------------------------+
            |                                  |
            | keyframes/ + final_cloud.ply     | (manual)
            v                                  v
            +--------> lidar_mesh_reconstruct.py
                       (.venv_o3d)
                       --> mesh.ply + mesh.obj
```

## 1. Producer (`lidar_textured_producer.py`)

Corre en el venv con **`unitree_webrtc_connect`** (de
[legion1581/unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect))
+ `aiortc` + `opencv` (en este proyecto: **`.venv`**).

> **Importante**: hay dos libs parecidas en circulacion. La correcta para este
> pipeline es `unitree_webrtc_connect` (clase `UnitreeWebRTCConnection`), NO
> `go2_webrtc_driver` del fork mas viejo. Si tenes la otra instalada, sacala:
> `pip uninstall go2-webrtc-connect`.

Instalacion (Mac):
```bash
pip install unitree_webrtc_connect
# Linux ademas:  sudo apt install -y portaudio19-dev
```

Que hace:

- Conecta al Go2 por LocalAP (IP fija `192.168.12.1`, tu Mac tiene que estar
  asociada al WiFi del robot).
- **Llama `await conn.datachannel.disableTrafficSaving(True)`** — sin esto el
  Go2 no envia el voxel_map.
- Setea decoder `libvoxel` y activa el LiDAR (`publish_without_callback` a
  `rt/utlidar/switch` con `"on"`).
- Suscribe a `rt/utlidar/voxel_map_compressed`, `rt/lf/sportmodestate` y al
  track de video.
- Por cada frame LiDAR:
  1. Decodifica voxels -> XYZ en frame LiDAR.
  2. Proyecta los puntos en la imagen mas reciente -> RGB por punto.
  3. Aplica transform LiDAR -> world usando la pose del robot.
  4. Acumula en un mapa global voxel-hashed que **promedia color por celda**.
- Cada ~500 ms vuelca `live_cloud.npz` (atomic write) para el viewer.
- Cada vez que el robot se mueve >30 cm o gira >15 grados guarda un **keyframe**
  (imagen + pose + intrinsics) para el refinamiento offline.
- Al cerrar (Ctrl+C) guarda `final_cloud.ply`, `trajectory.json` y los
  keyframes.

Correr:

```bash
# 1) asociar el Mac al WiFi que arma el Go2 (SSID tipo Unitree-Go2-XXXX).
# 2) verificar que llegues a la IP del robot:
ping -c 1 192.168.12.1

# 3) arrancar el producer
source .venv/bin/activate
cd go2_webrtc_connect-master/examples/data_channel/lidar
python lidar_textured_producer.py --output-dir scan_output
```

Flags utiles (`--help` para todos):

- `--voxel-size 0.04`           tamano de celda del mapa global (m)
- `--max-range 10.0`            corte de distancia LiDAR
- `--fx --fy --cx --cy`         intrinsics camara
- `--cam-dx --cam-dy --cam-dz`  posicion de la camara expresada en el frame del
                                 LiDAR (forward / left / up). Defaults pensados
                                 para el Go2: (+0.04, 0, -0.08) m.
- `--no-save-keyframes`         si no queres guardar keyframes a disco

Tips de calibracion:

- Si los colores quedan corridos respecto a las superficies, jugar con
  `--cam-dx/dy/dz` o con `--fx/fy/cx/cy`. Para Go2 con el stream WebRTC el FOV
  efectivo ronda 110 grados horizontal.
- Si la camara entrega 640x360 en lugar de 1280x720, pasar `--cam-width 640
  --cam-height 360 --fx 210 --fy 210 --cx 320 --cy 180`.

## 2. Viewer en vivo (`lidar_textured_viewer.py`)

Corre en `.venv_o3d` (tiene Open3D).

Polea el archivo `live_cloud.npz` y muestra el cloud coloreado. Atajos:

| Tecla | Accion                                          |
|-------|--------------------------------------------------|
| `R`   | reconstruir malla Poisson (toggle cloud / mesh) |
| `T`   | mostrar / ocultar la trayectoria del robot      |
| `C`   | ciclar modo de color: camara / altura / blend   |
| `N`   | alternar normales visibles                      |
| `O`   | alternar filtro de outliers (statistical)       |
| `S`   | guardar PNG de la ventana                       |
| `+/-` | tamano de punto                                 |
| `Q`   | salir                                            |

```bash
source .venv_o3d/bin/activate
cd go2_webrtc_connect-master/examples/data_channel/lidar
python lidar_textured_viewer.py --input scan_output/live_cloud.npz
```

## 3. Reconstruccion offline (`lidar_mesh_reconstruct.py`)

Corre en `.venv_o3d`. Toma el `final_cloud.ply` y produce una malla cerrada con
colores por vertice. Si le pasas la carpeta `keyframes/`, hace un segundo
pase de **multi-view color refinement** que mejora notablemente la textura.

```bash
source .venv_o3d/bin/activate
python lidar_mesh_reconstruct.py \
    --input scan_output/final_cloud.ply \
    --keyframes scan_output/keyframes \
    --output-prefix scan_output/mesh \
    --depth 10 \
    --voxel-size 0.025
```

Sale:

- `scan_output/mesh.ply` (con colores por vertice)
- `scan_output/mesh.obj`

Para abrir en Blender, MeshLab, CloudCompare o cualquier viewer 3D.

### Tuning

- `--depth`: 8 = rapido y suave, 10 = detalle alto (mas tiempo y RAM), 11 = bestia.
- `--density-quantile`: subir (0.06-0.10) para podar mas superficies espureas.
- `--voxel-size`: 0.02-0.03 m para indoor; 0.05+ para outdoor / mapas grandes.
- `--keyframe-blend`: 0.0 mantiene el color original, 1.0 usa solo el refinado.

## Workflow tipico

1. Encender el Go2, ponerlo en modo deportivo.
2. `python lidar_textured_producer.py --output-dir scan_output` (terminal A,
   `.venv`).
3. `python lidar_textured_viewer.py --input scan_output/live_cloud.npz`
   (terminal B, `.venv_o3d`).
4. Caminar despacio con el robot por el ambiente. Movimientos suaves,
   giros lentos. Pasar dos veces por las zonas importantes ayuda a estabilizar
   colores y normales.
5. Ctrl+C en el productor cuando termines.
6. `python lidar_mesh_reconstruct.py --input scan_output/final_cloud.ply
   --keyframes scan_output/keyframes`.
7. Abrir el `.obj` en MeshLab/Blender.

## Como mejora respecto al script original

| Aspecto                | `lidar_stream.py` original | Este pipeline                                              |
|------------------------|----------------------------|------------------------------------------------------------|
| Coloreo                | Ninguno (solo imprime)     | Color real de camara por punto, blendeado en cada celda    |
| Pose                   | Ignorada                   | LiDAR->world usando `sportmodestate` (position + RPY)      |
| Acumulacion            | Inexistente                | Voxel hash con promedio de color y conteo de muestras      |
| Outliers               | Sin filtrado               | Filtro estadistico online + offline                        |
| Estructuras            | No se ven                  | Reconstruccion Poisson con normales orientadas             |
| Texturizado            | No                         | Color por vertice + refinamiento multi-view por keyframes  |
| Visualizacion          | `print` en consola         | Open3D en vivo con toggles de mesh / colores / trayectoria |
| Persistencia           | Ninguna                    | NPZ live + PLY final + keyframes + trajectory.json + OBJ   |
