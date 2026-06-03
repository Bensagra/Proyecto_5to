# 🐕 Unitree Go2 - Dog Follower MEJORADO

## ✨ Cambios Principales

### 🔄 Sistema de Movimiento RECONSTRUIDO

El sistema de movimiento ha sido completamente reescrito con una lógica más robusta:

#### **Antes (seguimiento_solo.py):**
- Control básico sin integración de sensores
- Variables globales para estado
- Suavizado simple
- Sin evitación de obstáculos

#### **Ahora (seguimiento_perro_mejorado.py):**

```
┌─────────────────────────────────┐
│    ENTRADA: frame + LiDAR       │
├─────────────────────────────────┤
│  ↓ Detector SSD (persona)      │
│  ↓ Calcular error horizontal   │
│  ↓ Control PD para giro        │
│  ↓ Control PD para avance      │
│  ↓ Analizar LiDAR obstáculos   │
│  ↓ Suavizar comandos finales   │
├─────────────────────────────────┤
│  SALIDA: vx, vy, vz suavizado  │
└─────────────────────────────────┘
```

### 🎯 Mejoras en el Control

1. **Control PD mejorado:**
   - Curvas de control más suave (potencia 1.7)
   - Penalización de velocidad según ángulo de giro
   - Histéresis para evitar "serpenteo"
   - Suavizado exponencial de errores

2. **Integración LiDAR:**
   - Análisis de sector frontal (±30°)
   - 3 estados: `clear` → `slow` → `emergency`
   - Reduce velocidad cuando hay obstáculos cercanos
   - Detiene completamente a <45cm

3. **Suavizado final:**
   - Suavizado exponencial de comandos (alpha=0.18)
   - Evita cambios bruscos de velocidad
   - Movimientos más naturales

### 📊 Estructura Mejorada

```
Config                    → Todos los parámetros en un lugar
PersonDetector           → SSD con GPU/CPU automático
FaceMemory              → Reconocimiento de caras
FaceCropper             → Extracción de ROI
LidarAnalyzer           → Análisis de obstáculos ✅ NUEVO
ImprovedFollowController → Controlador PD ✅ RECONSTRUIDO
RobotCommandSender      → Envío seguro de comandos
```

---

## 🚀 Uso

### Instalación de dependencias
```bash
pip install unitree_webrtc_connect opencv-python numpy torch torchvision
```

### Uso básico
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181
```

### Con LiDAR
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181
```

### Sin LiDAR (más rápido)
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 --no-lidar
```

### LocalAP en lugar de STA
```bash
python seguimiento_perro_mejorado.py --ap
```

### Con Serial Number
```bash
python seguimiento_perro_mejorado.py --serial B42D2000P7I9GF8A
```

### Modo Debug (verbose)
```bash
python seguimiento_perro_mejorado.py --ip 192.168.8.181 --debug
```

---

## ⚙️ Configuración (archivo Config)

### Parámetros principales:

```python
# ── Detectar ─────────────────────────────────
confidence_threshold = 0.55      # umbral de confianza SSD
desired_box_width_ratio = 0.24   # ratio de ancho deseado
stop_box_width_ratio = 0.42      # ratio para "muy cerca"

# ── Velocidades ──────────────────────────────
max_forward_speed = 2.0          # m/s máximo adelante
min_forward_speed = 1.0          # m/s mínimo
max_turn_speed = 0.6             # rad/s máximo giro

# ── Control PD ───────────────────────────────
kp_forward = 10.0                # ganancia proporcional distancia
kp_turn = 0.8                    # ganancia proporcional giro
smoothing_alpha = 0.18           # suavizado (0-1, más alto=suave)
smooth_error_alpha = 0.75        # suavizado del error medido

# ── LiDAR ────────────────────────────────────
lidar_emergency_stop_m = 0.45    # espacio mínimo para emergencia
lidar_slow_zone_m = 0.90         # zona de reducción de velocidad
lidar_front_angle_deg = 60.0     # cono analizado (±30°)
lidar_height_min = -0.15         # filtro altura mínima
lidar_height_max = 1.80          # filtro altura máxima

# ── Caras ────────────────────────────────────
save_face_crops = True           # guardar caras nuevas
face_similarity_threshold = 0.6  # umbral de similitud

# ── Timeouts ─────────────────────────────────
person_lost_timeout_s = 0.5      # segundos sin detección
command_interval = 0.04          # Hz del control loop
```

---

## 🎮 Controles en tiempo real

| Tecla   | Acción                          |
|---------|--------------------------------|
| `Q`     | Salir del programa             |
| `S`     | Parada manual (emergency stop) |
| `N`     | Reestablecer modo normal robot |

---

## 📊 Estados del Robot

| Estado              | Descripción                                    |
|-------------------|----------------------------------------------|
| `IDLE`            | Sin target detectado                          |
| `HOLD`            | Target a distancia correcta, esperar          |
| `SLOW_FOLLOW`     | Seguir lentamente                            |
| `FAST_FOLLOW`     | Seguir con velocidad media                   |
| `RUN`             | Correr hacia el target (muy lejos)           |
| `TOO_CLOSE`       | Target muy cerca, detener                    |
| `BACK_OFF`        | Retroceder levemente (excesivamente cerca)   |
| `+LIDAR_SLOW`     | Obstáculo cercano, reducir velocidad         |
| `+LIDAR_EMERGENCY`| Obstáculo MUY cercano, PARAR TODO            |

---

## 📈 Visualización en pantalla

```
┌─────────────────────────────────────────────────────┐
│ FPS: 28.5                                           │
│ People: 2                                           │
│ CMD: x=+0.85 z=+0.12 box=0.32 mode=FAST_FOLLOW    │
│ LIDAR: 1.23m [slow]                                 │
│                                                     │
│ [Detecciones en verde]                              │
│ [Target en cyan]                                    │
│ [Caras en azul]                                     │
│ [Línea central en amarillo]                         │
└─────────────────────────────────────────────────────┘
```

---

## 🔍 Datos de LiDAR

El sistema recibe puntos cloud 4D:
```python
positions: list[float]  # [x, y, z, x, y, z, ...]  (en metros)
```

**Sistema de coordenadas Go2:**
- **X** = adelante/atrás (positivo = adelante)
- **Y** = izquierda/derecha (positivo = izquierda)
- **Z** = arriba/abajo (positivo = arriba)

**Análisis realizado:**
1. Filtro por altura: -0.15m a 1.80m
2. Calcular distancia 2D (√(x² + y²))
3. Calcular ángulo (atan2(y, x))
4. Sector frontal: ±30° (por defecto)
5. Detectar obstáculos segun distancia

---

## 🧠 Lógica de Control (reconstruida)

### 1️⃣ Error Horizontal
```
err_x = (centro_bbox - centro_pantalla) / ancho_pantalla
err_x suavizado = (error_previo * 0.75) + (error_nuevo * 0.25)
```

### 2️⃣ Cálculo de Giro
```
Si |err_x| > zona_muerta (0.035):
    norm_err = (|err_x| - zona_muerta) / (1 - zona_muerta)
    turn_strength = norm_err ^ 1.7  (curva suave)
    z_mag = 0.10 + turn_strength * (max_turn - 0.10)
    z = sign(err_x) * z_mag * kp_turn
Sino:
    z = 0
```

### 3️⃣ Cálculo de Velocidad
```
distance_error = desired_box_width - real_box_width

Si distance_error > 0.12:
    modo = "RUN"
    x = clamp(kp_forward * distance_error, 1.2, 2.0)
Sino si distance_error > 0.08:
    modo = "FAST_FOLLOW"
    ...
(etc)

Aplicar penalización por giro:
    Si |err_x| < 0.10: penalty = 1.0
    Si |err_x| < 0.22: penalty = 0.88
    Si |err_x| < 0.35: penalty = 0.72
    Sino: penalty = 0.50
    x *= penalty
```

### 4️⃣ Integración LiDAR
```
Si obstacle_state == "emergency":
    x = 0, z = 0  (PARAR TODO)
Sino si obstacle_state == "slow":
    x *= 0.5  (reducir velocidad)
```

### 5️⃣ Suavizado Final
```
x_smooth = prev_x * 0.18 + x * 0.82
y_smooth = prev_y * 0.18 + y * 0.82
z_smooth = prev_z * 0.18 + z * 0.82
```

---

## 🐛 Troubleshooting

### El robot no sigue correctamente
1. Ajusta `confidence_threshold` más alto si hay falsos positivos
2. Ajusta `kp_forward` y `kp_turn` para hacer control más/menos agresivo
3. Aumenta `smoothing_alpha` para movimientos más suaves

### Gira en la dirección opuesta
- Cambiar `invert_turn = True` en Config

### Imagen espejada
- Cambiar `mirror_image = True` en Config

### Demasiada oscilación
- Aumentar `smooth_error_alpha` (hacia 1.0)
- Aumentar `smoothing_alpha` (hacia 1.0)

### LiDAR activa paradas de emergencia constantes
- Aumentar `lidar_emergency_stop_m` (0.45 → 0.60)
- O desactivar con `--no-lidar`

---

## 📝 Comparación: Antes vs Después

| Aspecto                | Antes                | Después               |
|----------------------|-------------------|------------------|
| Sistema de movimiento | Básico            | PD reconstruido  |
| LiDAR                | No                | ✓ Sí, integrado  |
| Detección obstáculos | No                | ✓ Sí             |
| Suavizado            | Simple            | Exponencial      |
| Organización código  | Variables globales| Clases + Config  |
| Ganancias PD         | Hardcodeadas      | Configurables    |
| Histéresis giro      | Básica            | Mejorada         |
| Threading           | Básico            | Asyncio + Worker |
| Documentación       | Minimal           | Completa         |

---

## 💡 Tips Avanzados

### Para seguimiento agresivo (perro rápido):
```python
max_forward_speed = 2.5
kp_forward = 12.0
smoothing_alpha = 0.10  # más responsivo
```

### Para seguimiento suave (niño pequeño):
```python
max_forward_speed = 1.0
min_forward_speed = 0.5
kp_forward = 6.0
smoothing_alpha = 0.30  # más suave
```

### Para entorno con muchos obstáculos:
```python
lidar_slow_zone_m = 1.20
lidar_emergency_stop_m = 0.60
```

---

## 📦 Archivos

- **seguimiento_perro_mejorado.py** - Código principal (este archivo)
- **capturas_caras/** - Directorio donde se guardan las caras detectadas

---

## 🔗 Referencias

- Librería: `unitree_webrtc_connect`
- Modelo: SSD Lite MobileNet v3 (Torchvision)
- LiDAR: Go2 4D (PointCloud + tiempo real)

---

**Hecho con ❤️ para el Unitree Go2**
