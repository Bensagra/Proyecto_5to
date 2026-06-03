# Proyecto 5to

Repositorio de pruebas y desarrollo para control, seguimiento y vision por computadora con Unitree Go2. Incluye ejemplos de conexion WebRTC, seguimiento de personas/caras, lectura de LiDAR y scripts para entrenar modelos de deteccion de humanos.

## Contenido

- `go2_webrtc_connect-master/`: driver y ejemplos principales para conectar con el robot.
- `seguimiento_perro_mejorado.py`: seguimiento con deteccion visual.
- `seguimiento_perro_mejorado_fixed.py`: variante corregida del flujo de seguimiento.
- `test_conexion.py`: prueba rapida de conexion.
- `train_yolov8_model.py`: entrenamiento de detector con YOLOv8.
- `train_detection_model.py`: entrenamiento con Faster R-CNN.
- `predict.py`: predicciones con modelos entrenados.
- `README_ENTRENAMIENTO.md` y `TRAINING_GUIDE.md`: guias especificas de entrenamiento.

## Instalacion

Crear un entorno virtual local e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para los ejemplos que usan `unitree_webrtc_connect`, instalar tambien:

```bash
pip install unitree_webrtc_connect
```

Algunos scripts de LiDAR/Open3D pueden requerir dependencias extra segun el ejemplo que se ejecute.

## Uso rapido

Probar conexion:

```bash
python test_conexion.py
```

Ejecutar seguimiento:

```bash
python seguimiento_perro_mejorado_fixed.py
```

Entrenar YOLOv8:

```bash
python train_yolov8_model.py
```

Hacer una prediccion:

```bash
python predict.py --source imagen.jpg --model-type yolo
```

## Datos y artefactos locales

No se versionan entornos virtuales, capturas de caras, datasets, checkpoints, pesos de modelos ni resultados de prediccion. Esos archivos pueden contener datos privados o pesar demasiado para GitHub.

Antes de correr entrenamiento, ajustar las rutas del dataset en los scripts o colocar los datos esperados localmente.

## Preparacion para GitHub

Estado recomendado antes de subir:

```bash
git status
git add .
git commit -m "Prepare project for GitHub"
git push origin main
```
