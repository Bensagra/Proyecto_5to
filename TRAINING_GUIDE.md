# Entrenamiento de Modelo de Detección de Objetos con PyTorch

Este proyecto contiene código para entrenar modelos de detección de humanos usando tu dataset.

## 📋 Dataset

- **Formato**: CSV con anotaciones (bounding boxes)
- **Imágenes**: 416x416 píxeles
- **Clases**: Human (1 clase)
- **Estructura**:
  - `train/` - Imágenes de entrenamiento
  - `valid/` - Imágenes de validación
  - `test/` - Imágenes de prueba
  - `_annotations.csv` - Anotaciones por split

## 🚀 Instalación

### 1. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# o: venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

## ⚙️ Opciones de Entrenamiento

### Opción 1: Faster R-CNN (Más potente)

**Ventajas**:
- Mejor precisión en detecciones
- Bueno para datasets pequeños/medianos
- Más control sobre la arquitectura

**Desventajas**:
- Más lento
- Requiere más memoria GPU

**Ejecutar**:
```bash
python train_detection_model.py
```

**Qué hace**:
1. Carga el dataset desde CSV
2. Entrena modelo Faster R-CNN ResNet50
3. Guarda checkpoints del mejor modelo
4. Realiza predicciones en imagen de test
5. Genera gráficos de pérdida

**Salida**:
```
checkpoints/
├── best_model_epoch_X.pth       # Mejor modelo
├── final_model.pth              # Modelo final
└── prediction_result.jpg         # Resultado visual
```

---

### Opción 2: YOLOv8 (Recomendado para velocidad)

**Ventajas**:
- ⚡ Mucho más rápido
- 🎯 Excelente precisión
- 📊 Dataset preprocessing automático
- 🔄 Augmentación incorporada
- 💾 Modelo más compacto

**Desventajas**:
- Requiere conversión de formato

**Ejecutar**:
```bash
python train_yolov8_model.py
```

**Qué hace**:
1. Convierte anotaciones CSV a formato YOLO (automático)
2. Entrena YOLOv8 nano (más rápido) o small/medium
3. Genera dataset con estructura YOLO
4. Guarda mejor modelo
5. Realiza predicciones

**Salida**:
```
yolo_dataset/               # Dataset convertido
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/

yolo_results/
└── human_detection/
    ├── weights/
    │   ├── best.pt         # Mejor modelo
    │   └── last.pt
    └── results.csv

predictions/                # Imágenes con detecciones
```

---

## 📊 Comparativa

| Característica | Faster R-CNN | YOLOv8 |
|---|---|---|
| Velocidad | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Precisión | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Facilidad uso | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memoria requerida | Alta | Media |
| Tiempo entrenamiento | 📈 Lento | ⚡ Rápido |
| Modelos disponibles | 1 | 5 (nano-large) |

**Recomendación**: YOLOv8 para la mayoría de casos. Faster R-CNN si necesitas máxima precisión.

---

## 🎛️ Ajuste de Hiperparámetros

### Para Faster R-CNN (train_detection_model.py):
```python
BATCH_SIZE = 8          # Aumentar si hay GPU memory
NUM_EPOCHS = 20         # Ciclos de entrenamiento
LEARNING_RATE = 0.001   # Velocidad de aprendizaje
```

### Para YOLOv8 (train_yolov8_model.py):
```python
epochs=20           # Ciclos
imgsz=416          # Tamaño imagen (416, 640)
batch=8            # Batch size
patience=10        # Early stopping
optimizer='Adam'   # SGD, Adam, AdamW
lr0=0.001         # Learning rate inicial
```

---

## 💾 Cargar Modelo Entrenado

### Faster R-CNN:
```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(num_classes=2)
checkpoint = torch.load("checkpoints/best_model_epoch_X.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### YOLOv8:
```python
from ultralytics import YOLO

model = YOLO('yolo_results/human_detection/weights/best.pt')
results = model.predict(source='imagen.jpg', conf=0.5)
```

---

## 📈 Monitoreo del Entrenamiento

- **Logs en terminal**: Pérdida por época
- **Mejores checkpoints**: Se guardan automáticamente
- **Gráficos**: YOLOv8 genera automáticamente

---

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"
- Reduce `BATCH_SIZE` (8 → 4 o 2)
- Usa YOLOv8 nano en lugar de large

### Las predicciones son pobres
- Aumenta `NUM_EPOCHS` (20 → 50)
- Ajusta `LEARNING_RATE` (reduce si oscila)
- Verifica anotaciones en CSV

### Imágenes no se cargan
- Verifica ruta en `DATASET_PATH`
- Comprueba formato: .jpg, .jpeg, .png

### GPU no se detecta
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 Notas sobre tu Dataset

Tu dataset tiene:
- ✓ **1 clase**: Human
- ✓ **Formato**: Bounding boxes (xmin, ymin, xmax, ymax)
- ✓ **Anotaciones**: En CSV con columnas estándar
- ✓ **Tamaño imagen**: 416x416 (óptimo para YOLO)
- ✓ **Splits**: Train/valid/test

**Recomendaciones**:
1. Asegúrate de que los valores de bounding box están dentro de [0, width) y [0, height)
2. Verifica que no haya anotaciones vacías
3. Considera aumentar datos si el dataset es pequeño

---

## 🎯 Siguientes Pasos

1. **Entrenar modelo**: Elige YOLOv8 o Faster R-CNN
2. **Evaluar resultados**: Revisa métricas en logs
3. **Optimizar**: Ajusta hiperparámetros según resultados
4. **Desplegar**: Convierte modelo a formato para producción (ONNX, TorchScript)

---

## 📚 Referencias

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Faster R-CNN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- [YOLO Format](https://docs.ultralytics.com/datasets/detect/)

---

## 💡 Tips

- **Early Stopping**: YOLOv8 tiene `patience` incorporado
- **Data Augmentation**: YOLOv8 aplica automáticamente, Faster R-CNN necesita code adicional
- **Validación Real-time**: Monitorea métricas cada epoch
- **Exportar Modelo**: Ambos soportan exportación a ONNX, TorchScript, etc.

