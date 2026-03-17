# 🎮 CONTROL DEL GO2 - SIN CÁMARA, SOLO FLECHAS

He investigado los archivos y creado **dos versiones** para controlar el Go2:

## 📊 Investigación Realizada

### Estructura de Comunicación:
```python
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
await conn.connect()

# Enviar comando de movimiento:
move_cmd = {
    "api_id": SPORT_CMD["Move"],      # ID 1008
    "parameter": {
        "x": vx,          # Forward/backward (-1.0 a 1.0)
        "y": vy,          # Left/right (-0.5 a 0.5)
        "z": 0
    }
}

response = await conn.datachannel.pub_sub.publish_request_new(
    RTC_TOPIC["SPORT_MOD"],
    move_cmd,
    timeout=5
)
```

### Basado en:
- `sportmode.py`: Patrón de conexión y comandos
- `teleop_arrows.py`: Waitkey y manejo de teclado
- `go2_webrtc_driver/constants.py`: SPORT_CMD, RTC_TOPIC

---

## 🎮 Versión 1: `control_simple.py`

**Características:**
- ✅ **Entrada por terminal** (input bloqueante)
- ✅ **Sin GUI ni OpenCV**
- ✅ **Muy simple**
- ❌ No es tiempo real (bloqueante)

**Uso:**
```bash
python control_simple.py

>> w      # Adelante
>> a      # Izquierda
>> space  # Detener
>> q      # Salir
```

**Ventajas:**
- Minimal, sin dependencias extra
- Bueno para scripts
- Fácil de entender

**Desventajas:**
- Debe presionar ENTER después de cada tecla
- No es interactivo en tiempo real

---

## ⚡ Versión 2: `control_flechas.py` (RECOMENDADO)

**Características:**
- ✅ **Captura DIRECTA de flechas** (tiempo real)
- ✅ **Soporta flechas + WASD**
- ✅ **Window mínima con estado**
- ✅ **~10 Hz de envío de comandos**
- ✅ **Aceleración progresiva**

**Uso:**
```bash
python control_flechas.py
```

Luego simplemente presiona:
```
↑ / W   → Adelante
↓ / S   → Atrás
← / A   → Izquierda
→ / D   → Derecha
ESPACIO → Detener
Q       → Salir
```

**Interfaz:**
```
╔════════════════════════════════╗
║      GO2 CONTROL              ║
║  Vx: +0.50 | Vy: -0.30       ║
║  Presiona Q para salir        ║
╚════════════════════════════════╝
```

**Ventajas:**
- Tiempo real con flechas
- Sin necesidad de presionar ENTER
- Mejor UX para teleoperación

---

## 📊 Comparativa

| Feature | control_simple.py | control_flechas.py |
|---------|---|---|
| **Input** | Terminal bloqueante | Teclado tiempo real |
| **Flechas** | Solo wasd | Flechas + WASD ✓ |
| **GUI** | Ninguna | Ventana mínima |
| **Hz** | Variable | ~10 Hz constante |
| **Aceleración** | Sí | Sí |
| **Complejidad** | Baja | Media |

---

## 🚀 Recomendación

**Para controlar el Go2 en tiempo real:** Usa **`control_flechas.py`**

```bash
python control_flechas.py
```

Es la más responsiva y exacta.

---

## ⚙️ Parámetros Configurables

En ambos archivos:

```python
MAX_SPEED_X = 1.0      # Velocidad máxima adelante/atrás
MAX_SPEED_Y = 0.5      # Velocidad máxima giro
SPEED_INCREMENT = 0.1  # Cuánto acelera por teclazo
```

### Ejemplo: Control más sensible
```python
SPEED_INCREMENT = 0.2  # Responde más rápido
MAX_SPEED_X = 0.7      # Límite menor
```

---

## 🔍 Diferencias con `teleop_arrows.py`

| Feature | control_flechas.py | teleop_arrows.py |
|---------|---|---|
| **Cámara** | ❌ NO | ✅ SÍ |
| **Complejidad** | ⭐ Baja | ⭐⭐⭐ Alta |
| **Velocidad** | Rápido | +Lento (procesa video) |
| **FPS** | N/A | ~10-15 FPS |
| **Reconexión** | Simple | Robusta |

Para solo **mover el robot** → **`control_flechas.py`** es perfecto  
Para **seguimiento + cámara** → **`teleop_arrows.py`**

---

## 🛠️ Cómo Funciona Internamente

### control_flechas.py:

```
Main Thread (UI + Teclado)
    ↓
Detecta tecla (arrow keys)
    ↓
Actualiza RobotControl.vx, .vy
    ↓
Dibuja estado en ventana
    ↓
────────────────────────────────
    ↓
AsyncIO Thread (Comandos)
    ↓
Cada ~100ms:
  Crea comando Move(vx, vy)
    ↓
  Envía por datachannel
    ↓
  Robot recibe y ejecuta
```

---

## 📝 Notas Importantes

1. **Datachannel**: Espera hasta 20 segundos a que esté listo (LocalAP puede ser lento)
2. **Timeout**: 5 segundos por comando (para evitar bloqueos)
3. **Deadman**: Si no presionas tecla, el robot NO se detiene automáticamente (puedes enviar ESPACIO para detener)
4. **Hz**: Envía ~10 comandos/segundo (suficiente para suave)

---

## 🎯 Quick Start

```bash
# Opción 1: Tiempo real con flechas (RECOMENDADO)
python control_flechas.py

# Opción 2: Terminal simple
python control_simple.py
```

¡Listo! 🚀
