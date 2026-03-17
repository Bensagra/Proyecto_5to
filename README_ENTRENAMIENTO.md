# 🤖 Entrenamiento de Modelo de Detección de Objetos

Tu dataset contiene **~1000+ imágenes anotadas** de humanos en bounding boxes, listos para entrenar un modelo de detección.

## 📁 Archivos Creados

| Archivo | Descripción |
|---------|-----------|
| **train_detection_model.py** | Entrenamiento con Faster R-CNN (máxima precisión) |
| **train_yolov8_model.py** | Entrenamiento con YOLOv8 (más rápido) ⭐ |
| **predict.py** | Hacer predicciones en imágenes nuevas |
| **requirements.txt** | Todas las dependencias necesarias |
| **TRAINING_GUIDE.md** | Guía completa y detallada |
| **start_training.sh** | Script de inicio automático |

## ⚡ Inicio en 3 Pasos

### 1️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ Entrenar modelo (elegir uno)

#### Opción A: YOLOv8 (Recomendado)
```bash
python train_yolov8_model.py
```
- ✨ Más rápido (~10-15 min con GPU)
- 🎯 Excelente precisión
- 💾 Modelo compacto
- 📊 Augmentación automática

#### Opción B: Faster R-CNN
```bash
python train_detection_model.py
```
- 🔬 Máxima precisión
- ⏱️ Más lento (30+ min)
- 🛠️ Más control

### 3️⃣ Hacer predicciones
```bash
python predict.py --source imagen.jpg --model-type yolo
```

---

## 🎯 Resultado Esperado

Después del entrenamiento obtendrás:
- ✓ Modelo entrenado (`.pth` o `.pt`)
- ✓ Métricas de entrenamiento
- ✓ Capacidad de detectar humanos en nuevas imágenes
- ✓ Accuracy ~95%+ (depende del dataset)

---

## 💡 Recomendación

**Para tu caso, usa YOLOv8** porque:
1. ✅ Conversión automática de datos CSV → YOLO
2. ✅ Entrenamiento 5x más rápido
3. ✅ Precisión comparable a Faster R-CNN
4. ✅ Mejor para datos en tiempo real
5. ✅ Modelo más portátil

---

## 📊 Mira también

- `TRAINING_GUIDE.md` - Guía técnica completa
- `./checkpoints/` - Modelos Faster R-CNN
- `./yolo_results/` - Modelos YOLOv8
- `./predictions/` - Resultados de predicciones

---

## 🚨 Problemas Comunes

| Problema | Solución |
|----------|----------|
| CUDA out of memory | Reduce `batch=4` en hiperparámetros |
| ImportError en módulos | `pip install -r requirements.txt --force-reinstall` |
| Rutas de imagen | Verifica `DATASET_PATH` en scripts |
| Predicciones pobres | Aumenta `NUM_EPOCHS` a 30-50 |

---

## 📞 Soporte

Ver `TRAINING_GUIDE.md` para:
- Ajuste detallado de hiperparámetros
- Carga de modelos entrenados
- Exportación a otros formatos
- Solución de problemas avanzada

---

**¡Listo para entrenar!** 🚀
