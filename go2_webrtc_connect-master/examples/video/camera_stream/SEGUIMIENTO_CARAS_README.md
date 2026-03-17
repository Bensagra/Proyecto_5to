# 👤 SEGUIMIENTO AUTÓNOMO CON DETECCIÓN DE CARAS - Go2

Script que combina:
- ✅ Detección de personas (SSD)
- ✅ Detección de caras (Haar Cascade)
- ✅ Tracking de personas (Centroid Tracker)
- ✅ Seguimiento autónomo del robot
- ✅ Guardado de caras nuevas

## 🚀 Cómo Usar

### 1. Configurar Conexión

```python
# LocalAP (default - recomendado)
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

# O LocalSTA con IP
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
```

### 2. Ejecutar

```bash
python seguimiento.py
```

### 3. Descripción del Flujo

```
┌─────────────────────────────────────┐
│  Frame del robot Go2 (1280x720)     │
└──────────────┬──────────────────────┘
               ↓
    ┌──────────────────────┐
    │  Detección SSD       │ → Lista de personas
    ├──────────────────────┤
    │  Centroid Tracker    │ → {ID: centroid}
    ├──────────────────────┤
    │  Detección Caras     │
    │  en cada persona     │
    └──────┬───────────────┘
           ↓
    ¿Cara Nueva?
    ├─→ SÍ → Guardar
    │        Auto-seleccionar persona
    │        Enviar comandos seguimiento
    └─→ NO → Continuar

           ↓
    ┌──────────────────────┐
    │   Comandos de Mov.   │
    │   al SPORT_MOD       │
    └──────────────────────┘
```

## 📊 Componentes

### SSDPersonDetector
- Detecta personas en el frame
- Confianza: 0.55
- Retorna: `[{"box": [x1,y1,x2,y2], "score": 0.95}, ...]`

### FaceCropper
- Detecta caras dentro de cada persona
- Usa Haar Cascade (rápido)
- Retorna: cara recortada + bbox global

### FaceMemory
- Almacena embeddings de caras vistas
- Compara similitud con nuevas caras
- Si distancia < threshold → cara repetida

### CentroidTracker
- Asigna ID único a cada persona detectada
- Sigue personas entre frames
- Elimina IDs después de 10 frames sin detección

### MovementController
- Calcula velocidades (vx, vy) según posición
- Dead zone: ±50px (sin movimiento)
- Control suave con zonas de avance

## 🎯 Comportamiento

### Al Detectar una Cara Nueva:
1. Se guarda en `capturas_caras/`
2. Se asocia con la persona más cercana
3. El robot comienza a seguir a esa persona
4. Imprime: `👤 Siguiendo persona ID`

### Mientras Sigue:
- Mantiene a la persona centrada
- Avanza si está lejos
- Retrocede si está cerca
- Gira para centrar horizontalmente

### Si Pierde la Persona:
- Estado cambia a `LOST` (rojo)
- Robot se detiene
- Espera nueva detección

## ⚙️ Parámetros Configurables

```python
# Confianza de detección
CONFIDENCE_THRESHOLD = 0.55  # Personas
FACE_CONFIDENCE_THRESHOLD = 0.7  # Caras (no usado actualmente)

# Tracking
MAX_DISTANCE = 100  # Distancia máxima para asociar
MAX_FRAMES_WITHOUT_DETECTION = 10  # Frames antes de eliminar

# Control de movimiento
DEAD_ZONE = 50  # ±px sin mover
FORWARD_ZONE = 100  # ±px distancia óptima
MAX_SPEED_X = 0.5  # Velocidad máxima adelante/atrás
MAX_SPEED_Y = 0.3  # Velocidad máxima giro

# Guardado de caras
MIN_SECONDS_BETWEEN_SAVES = 2.0  # Mínimo entre guardados
```

## 📁 Archivos Generados

```
capturas_caras/
├── cara_20260317_120530_123.jpg
├── cara_20260317_120535_456.jpg
└── ... (una por cada cara nueva detectada)
```

## 🔧 Diferencias con Otros Scripts

| Feature | `seguimiento.py` | `seguimiento_autonomo.py` | `teleop_arrows.py` |
|---------|---|---|---|
| **Detección** | SSD + Caras | Solo SSD | N/A |
| **Seguimiento** | Centroid + Caras | Centroid | Manual |
| **Guardado** | Sí | No | No |
| **Control** | Automático | Automático | Manual |
| **Inicio** | Auto (cara nueva) | Auto (1ª persona) | Manual |

## 🐛 Troubleshooting

### "El datachannel no abrió a tiempo"
- Aumenta `max_wait` en `wait_for_datachannel()` a 30 segundos
- Verifica conexión WiFi LocalAP

### El robot no se mueve
- Verifica que se detecten personas (bounding boxes verdes)
- Revisa si aparece "Siguiendo persona" en consola
- Aumenta `MAX_SPEED_X` a 0.7

### No detecta caras
- Acércate más al robot
- Mejora iluminación
- Redondea la cara hacia la cámara

### Guarda muchas caras falsas
- Aumenta `similarity_threshold` de FaceMemory a 0.8
- Incrementa `MIN_SECONDS_BETWEEN_SAVES` a 5.0

### FPS bajo
- El embarcado es la detección SSD
- Reduces tamaño de frame (pero reduce calidad)
- O usa GPU si disponible

## 📈 Flujo de Ejecución Completo

```
1. Conectar Go2 (LocalAP)
   └─ Esperar 1s estabilización
   └─ Activar cámara
   └─ Esperar datachannel ready (máx 20s)

2. Loop principal:
   ├─ Recibir frame
   ├─ Detectar personas
   ├─ Trackear con ID
   ├─ Por cada persona:
   │  ├─ Detectar cara
   │  ├─ Si cara nueva:
   │  │  ├─ Guardar
   │  │  └─ Auto-seleccionar
   │  └─ Si es persona tracked:
   │     ├─ Calcular offset
   │     ├─ Generar comando move
   │     └─ Enviar a SPORT_MOD
   ├─ Dibujar UI
   └─ Mostrar en pantalla

3. Al presionar 'Q':
   ├─ Enviar StopMove
   ├─ Cerrar ventanas
   ├─ Salir
```

## 🎮 Controles

- **Q**: Salir
- Sin controles de teclado avanzados (solo SSD + Caras)

## 💡 Tips Avanzados

1. **Mejorar FaceMemory**:
   ```python
   similarity_threshold = 0.4  # Más exigente
   # vs
   similarity_threshold = 0.8  # Más tolerante
   ```

2. **Aumentar Sensibilidad**:
   ```python
   DEAD_ZONE = 30  # Más sensible
   MAX_SPEED_X = 1.0  # Más rápido
   ```

3. **Debug Mode**:
   ```python
   logging.basicConfig(level=logging.DEBUG)  # Ver más logs
   ```

4. **Multiples Personas**:
   Actualmente sigue la primera cara detectada. Para seguir múltiples:
   ```python
   # Mantener lista de personas siendo seguidas
   following_ids = [id1, id2, id3]
   # Enviar comando promedio de movimientos
   ```

---
**Estado**: ✅ Funcional  
**Versión**: 1.0  
**Fecha**: 2026-03
