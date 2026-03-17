# 🤖 SEGUIMIENTO AUTÓNOMO DE PERSONA - Go2 WebRTC

## Descripción
Script para seguimiento autónomo de una persona en tiempo real. El robot Unitree Go2 detecta personas, mantiene el tracking entre frames y envía comandos de movimiento para seguir a la persona seleccionada.

## ✨ Características Principales

### 1. **Detección de Personas (SSD)**
- Usa modelo `SSDLite320_MobileNet_V3_Large` (pre-entrenado en COCO)
- Threshold configurable: `CONFIDENCE_THRESHOLD = 0.55`
- Eficiente en CPU/GPU

### 2. **Centroid Tracking**
- Mantiene identidad de personas entre frames
- Usa distancia euclidiana para asociación
- Elimina tracks que desaparecen (`MAX_FRAMES_WITHOUT_DETECTION = 10`)
- **No requiere RED Neural de seguimiento** - funciona muy rápido

### 3. **Control de Movimiento Inteligente**
```
Dead Zone (±50px): El robot no se mueve si la persona está centrada
Forward Zone (±100px): Define distancia óptima de seguimiento

      Arriba (Lejana) → Avanzar
          ↓
    ↙ Girar Izq. ↓ Centro ↑ Girar Der. →
          ↑
    Abajo (Cercana) → Retroceder
```

### 4. **Interfaz Visual**
- Crosshair central para referencia
- Zonas de control dibujadas
- ID de personas tracked
- Información en tiempo real (offset X/Y, FPS, estado)

## 🎮 Cómo Usar

### Paso 1: Ejecutar
```bash
python seguimiento_autonomo.py
```

### Paso 2: Seleccionar Persona
- **Opción A (Automático)**: Se selecciona la primera persona detectada
- **Opción B (Manual)**: Haz click en una persona para seguirla

### Paso 3: El robot comienza a seguir
Estados posibles:
- 🟢 **TRACKING**: Persona en posición óptima, listo para seguir
- 🟠 **TURNING**: Girando para centrar a la persona
- 🟡 **MOVING_FORWARD**: Avanzando o retrocediendo para mantener distancia
- 🔴 **LOST**: Persona no detectada

### Salir
Presiona **'q'** para terminar

## ⚙️ Parámetros Configurables

### Detección
```python
CONFIDENCE_THRESHOLD = 0.55  # Aumentar = menos falsos positivos
```

### Tracking
```python
MAX_DISTANCE = 100                    # Distancia máxima para asociar detecciones
MAX_FRAMES_WITHOUT_DETECTION = 10     # Frames hasta eliminar track
```

### Control de Movimiento
```python
DEAD_ZONE = 50          # Rango de tolerancia (±px desde centro)
FORWARD_ZONE = 100      # Distancia óptima de seguimiento

SPEED_FORWARD = 0.5     # Velocidad adelante (0-1)
SPEED_BACKWARD = -0.3   # Velocidad atrás (0-1)
SPEED_TURN_LEFT = -0.3  # Giro izquierda
SPEED_TURN_RIGHT = 0.3  # Giro derecha
```

## 📊 Flujo de Datos

```
Frame del Go2 (1280x720)
        ↓
   [Detector SSD]
        ↓
   [Lista de boxes]
        ↓
   [Centroid Tracker]
        ↓
   [Dict {ID: centroid}]
        ↓
   [Movement Controller]
        ↓
   Calcula (vx, vy)
        ↓
   [conn.motion.move()]
        ↓
   Robot se mueve
```

## 🔧 Mejoras Respecto al Original

| Aspecto | Original | Nuevo |
|---------|----------|-------|
| **Tracking** | Solo detección frame-a-frame | Centroid Tracking con ID persistente |
| **Control** | Sin comandos de movimiento | Control PID-like con zonas muertas |
| **Selección** | Guarda todas las caras | Selecciona una persona para seguir |
| **Interfaz** | Básica | Información detallada, zonas visuales |
| **Continuidad** | Pierde personas entre frames | Mantiene track por ~10 frames |

## 🎯 Casos de Uso

1. **Seguimiento de Persona**: Robot sigue a una persona a través del espacio
2. **Guardia de Seguridad**: Robot patrulla siguiendo a una persona sospechosa
3. **Robot Asistente**: Sigue al usuario para entregar objetos
4. **Demostración**: Muestra capacidades autónomas del Go2

## 📝 Troubleshooting

### El robot no se mueve
- Verifica conexión WebRTC: `conn.motion.move()` puede fallar
- Aumenta `SPEED_FORWARD` y `SPEED_TURN_RIGHT`

### Pierde tracking frecuentemente
- Aumenta `MAX_DISTANCE` (ej: 150 en lugar de 100)
- Aumenta `MAX_FRAMES_WITHOUT_DETECTION` (ej: 15)
- Reduce `CONFIDENCE_THRESHOLD` a 0.50

### El robot es muy sensible
- Aumenta `DEAD_ZONE` (ej: 80 en lugar de 50)
- Reduce velocidades de giro y avance

### Muy lento
- Ya usa CPU - asegúrate de tener CUDA si disponible
- `USE_CUDA = torch.cuda.is_available()` auto-detecta

## 🚀 Trabajo Futuro (Posibles Mejoras)

```python
# 1. Kalman Filter para predicción más suave
from filterpy.kalman import KalmanFilter

# 2. Historial de movimiento para predicción
class PredictiveTracker:
    def predict_position(self, id, frames_ahead=5):
        # Extrapolar basado en velocidad

# 3. Control PID para movimiento más natural
class PIDController:
    def update(self, error):
        return self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative

# 4. Multi-persona: Seguir grupo o más de una persona
# 5. Detección de gesto: Seguir si la persona hace un gesto específico
# 6. Guardar video del seguimiento
```

## 📚 Dependencias Requeridas

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
unitree-webrtc-connect
aiortc
```

## 💡 Tips de Rendimiento

1. **FPS**: Ejecuta a ~10-15 FPS en CPU típica
2. **Latencia**: ~100-150ms desde detección a movimiento
3. **Memoria**: ~500MB RAM, GPU opcional
4. **Distancia**: Prueba a 2-3 metros del robot

---
**Autor**: Sistema de Seguimiento Autónomo v1.0
**Fecha**: 2026-03
