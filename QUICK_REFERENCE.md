# 🚀 Quick Start - Dog Follower Mejorado

## Inicio Rápido (3 pasos)

### 1. Instalación
```bash
cd /Users/bensagra/Documents/Proyecto_5to
pip install unitree_webrtc_connect opencv-python numpy torch torchvision
```

### 2. Lanzar (uso básico)
```bash
# Con IP del robot
python seguimiento_perro_mejorado.py --ip 192.168.8.181

# O con LocalAP
python seguimiento_perro_mejorado.py --ap

# O con Serial Number
python seguimiento_perro_mejorado.py --serial B42D2000P7I9GF8A
```

### 3. Controles
- **Q** = Salir
- **S** = Stop de emergencia
- **N** = Resetear modo normal

---

## Ejemplos de Uso

### Seguimiento rápido (perro activo)
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 \
  --max-forward-speed 2.5 \
  --kp-forward 12.0 \
  --smoothing 0.10
```

### Seguimiento suave (niño pequeño)
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 \
  --max-forward-speed 1.0 \
  --kp-forward 6.0 \
  --smoothing 0.30
```

### Sin LiDAR (más rápido)
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 --no-lidar
```

### Modo Debug (verbose)
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 --debug
```

### Sin guardar caras
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 --no-face-save
```

---

## Sintaxis de Línea de Comandos

```
python seguimiento_perro_mejorado.py [OPTIONS]

OPTIONS:
  --ip ADDR              Robot IP (default: 192.168.8.181)
  --serial SERIAL        Robot serial number
  --ap                   Use LocalAP instead of LocalSTA
  --debug                Verbose debug output
  --no-lidar             Disable LiDAR
  --no-face-save         Don't save face crops
  --confidence FLOAT     Detection confidence threshold (0.0-1.0)
  --kp-forward FLOAT     Forward control gain
  --kp-turn FLOAT        Turn control gain
  --smoothing FLOAT      Command smoothing (0-1, higher=smoother)
  --max-forward FLOAT    Maximum forward speed (m/s)
  --max-turn FLOAT       Maximum turn speed (rad/s)
```

---

## Diagnóstico Rápido

### ¿Se conecta al robot?
```python
# En Python:
from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod
conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
# Si no hay error, está OK
```

### ¿Detecta personas?
- Buscar en output: "People: X"
- Si "People: 0", aumenta `--confidence` a 0.40

### ¿Activa LiDAR?
- Buscar en output: "LIDAR: X.XXm [clear|slow|emergency]"
- Si no aparece, usar `--no-lidar`

### ¿Se mueve?
- Buscar: "CMD: x=... z=..." en debug
- Si x=0, z=0 siempre: revisar detección

---

## Ajuste Rápido de Parámetros

### Oscila demasiado (serpentea)
```
↓ Aumentar smoothing (0.18 → 0.25)
↑ Bajar kp-turn (0.8 → 0.5)
↑ Aumentar smooth-error-alpha (0.75 → 0.85)
```

### Muy lentos los movimientos
```
↑ Bajar smoothing (0.18 → 0.10)
↑ Aumentar kp-forward (10 → 15)
↑ Aumentar kp-turn (0.8 → 1.2)
```

### Gira en dirección opuesta
```
Config.invert_turn = True
```

### Imagen espejada
```
Config.mirror_image = True
```

### LiDAR activa paradas de emergencia
```
Config.lidar_emergency_stop_m = 0.60  (antes 0.45)
```

---

## Variables de Control Clave

```python
# En Config dataclass

# Detección
confidence_threshold = 0.55      # Más alto = menos falsos positivos
desired_box_width_ratio = 0.24   # Ratio deseado relativo a pantalla
stop_box_width_ratio = 0.42      # Ratio para "muy cerca"

# Velocidades
max_forward_speed = 2.0          # m/s máximo
max_turn_speed = 0.6             # rad/s máximo

# Control
kp_forward = 10.0                # Ganancia distancia (D/dt)
kp_turn = 0.8                    # Ganancia giro
smoothing_alpha = 0.18           # 0=instant, 1=muy lento

# LiDAR (m = metros)
lidar_emergency_stop_m = 0.45    # < esto = STOP
lidar_slow_zone_m = 0.90         # < esto = 50% velocity
lidar_front_angle_deg = 60.0     # ±30° cone
```

---

## Flujo de Datos Principal

```
Camera Frame (1280x720)
            ↓
     PersonDetector (SSD)
            ↓
      chooseTarget()
            ↓
    ComputeFollowCommand()
            ↓
   ├─ Error Horizontal (center bias)
   ├─ PD Giro (turn with hystéresis)
   ├─ PD Velocidad (forward with penalty)
   └─ LiDAR Obstacles (reduce/stop)
            ↓
      Smooth Commands (alpha=0.18)
            ↓
   RobotCommandSender.send_move()
            ↓
        Go2 Robot Moves
```

---

## Estructura de Archivos

```
Proyecto_5to/
├── seguimiento_perro_mejorado.py   ← NUEVO (código principal)
├── GUIA_MEJORADO.md                ← NUEVA (guía completa)
├── QUICK_REFERENCE.md              ← ESTE ARCHIVO
├── capturas_caras/                 ← Caras guardadas
├── seguimiento_solo.py             ← (original para referencia)
└── ... (otros archivos del proyecto)
```

---

## Estados del Robot en Tiempo Real

Ver en pantalla:
```
FPS: 28.5                                    ← Frames por segundo
People: 1                                    ← Personas detectadas
CMD: x=+0.85 z=+0.12 box=0.32 mode=FAST    ← Comando actual
LIDAR: 1.23m [slow]                         ← Estado LiDAR
```

**Modos:**
- `RUN` = lejos, correr
- `FAST_FOLLOW` = lejos, seguir rápido
- `SLOW_FOLLOW` = cerca, seguir lento
- `HOLD` = distancia perfecta, esperar
- `TOO_CLOSE` = muy cerca, parar
- `+LIDAR_SLOW` = hay obstáculo, reduced speed
- `+LIDAR_EMERGENCY` = obstáculo crítico, PARAR

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| No se conecta | Verificar IP/Serial, intentar LocalAP con `--ap` |
| No detecta personas | `--confidence 0.40`, mejor iluminación |
| Oscila (serpentea) | `--smoothing 0.25`, `--kp-turn 0.5` |
| Muy lento | `--smoothing 0.10`, `--kp-forward 15` |
| Gira al revés | `invert_turn=True` en Config |
| LiDAR para constantemente | `--no-lidar` o aumentar `lidar_emergency_stop_m` |
| GUI lag/FPS bajo | `--no-face-save`, menos tamaño frame |

---

## Performance Tips

```bash
# Si FPS < 20:
python seguimiento_perro_mejorado.py --ip 192.168.8.181 \
  --no-face-save \        # No procesar caras
  --no-lidar              # Sin análisis LiDAR

# Si FPS < 10:
# También reducir confidence threshold
  --confidence 0.70
```

---

## Desarrollo / Debugging

### Ver logs detallados:
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 --debug 2>&1 | tee datos.log
```

### Por frame:
- Presionar `s` para registrar comando actual
- Presionar `n` para resetear robot mode
- Presionar `q` para salir

### Modificar parámetros en runtime:
Editar `Config` dataclass directamente en el código (líneas ~140-200)

---

## Comparación con Original (seguimiento_solo.py)

| Característica | Original | Mejorado |
|---------------|----------|---------|
| LiDAR | ✗ No | ✓ Sí |
| Evitación obstáculos | ✗ No | ✓ Sí |
| Control PD estructurado | ✗ Global vars | ✓ Clase |
| Configuración | Hardcoded | Dataclass |
| Threading | Básico | Asyncio + Workers |
| Documentación | Minimal | Completa |
| Curvatura control | Lineal | Suave (^1.7) |
| Histéresis giro | Básica | Mejorada |

---

## Próximas Mejoras Posibles

1. **Visión avanzada:**
   - Multi-person tracking con centroid
   - Reconocimiento facial (face_recognition lib)
   - YOLOv8 en lugar de SSD

2. **LiDAR avanzado:**
   - Mapeo de obstáculos (occupancy grid)
   - Path planning A*
   - Wall-follow behavior

3. **Control avanzado:**
   - Kalman filter para suavizado de trayectoria
   - Trajectory prediction
   - Velocity ramps (aceleración gradual)

4. **UI:**
   - Web dashboard (Flask)
   - 3D visualization del LiDAR
   - Logging de sesiones

---

**¡Listo para seguir perros! 🐕**

Cualquier duda: revisar `GUIA_MEJORADO.md` para referencia completa.
