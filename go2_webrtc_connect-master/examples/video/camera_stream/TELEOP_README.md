# 🎮 TELEOPERACIÓN CON FLECHAS - Go2 WebRTC

Control en tiempo real del robot Unitree Go2 usando las teclas de flecha del teclado.

## ✨ Características

✅ **Control en tiempo real** (~10 Hz)  
✅ **Flechas del teclado** para movimiento (o WASD)  
✅ **Aceleración progresiva** - mantén presionado para ir más rápido  
✅ **Deadman timeout** - se detiene automáticamente si no hay input  
✅ **Visualización en vivo** - velocidades y estado en pantalla  
✅ **Gráficos de velocidad** - barras visuales en tiempo real  

## 🚀 Cómo Usar

### 1. Configurar Conexión

Edita las primeras líneas de `teleop_arrows.py`:

```python
# Opción 1: LocalAP (recomendado)
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

# Opción 2: LocalSTA con IP
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")

# Opción 3: LocalSTA con Serial
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000P7I9GF8A")

# Opción 4: Remote (requiere credenciales)
conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, 
                          serialNumber="B42D2000P7I9GF8A",
                          username="email@gmail.com", 
                          password="password")
```

### 2. Ejecutar

```bash
python teleop_arrows.py
```

### 3. Controlar el Robot

```
↑ o W  → Adelante (aceleracion progresiva)
↓ o S  → Atrás (aceleracion progresiva)
← o A  → Girar izquierda
→ o D  → Girar derecha
Q     → Salir
```

## ⚙️ Parámetros Configurables

```python
# Velocidades máximas
MAX_SPEED_X = 1.0      # Adelante/atrás (-1.0 a 1.0)
MAX_SPEED_Y = 0.5      # Izquierda/derecha (-0.5 a 0.5)
SPEED_INCREMENT = 0.1  # Incremento por cada tecla presionada
```

### Ejemplo: Control más responsivo
```python
SPEED_INCREMENT = 0.2  # Respuesta más rápida
MAX_SPEED_X = 0.7      # Movimiento más lento
```

## 🔧 Cómo Funciona

### 1. Threading
- **Thread 1 (Asyncio)**: Conexión WebRTC, envío de comandos (~10 Hz)
- **Thread 2 (Main)**: OpenCV, captura de input del teclado (~20 Hz)

### 2. Deadman Timeout
Si no presionas una tecla en 0.3 segundos, el robot se detiene automáticamente:
```python
robot_state.apply_deadman(timeout=0.3)
```

### 3. Aceleración Progresiva
- Cada tecla presionada incrementa la velocidad en `SPEED_INCREMENT`
- Mantén presionado para acelerar hasta `MAX_SPEED_X/Y`
- Soltar la tecla NO detiene inmediatamente (deadman timeout se encarga)

### 4. Comandos al Robot
```python
# Cada 100ms se envía:
move_cmd = {
    "api_id": SPORT_CMD["Move"],
    "parameter": {
        "x": vx,    # Forward/backward (-1.0 a 1.0)
        "y": vy,    # Left/right (-0.5 a 0.5)  
        "z": 0      # (no usado)
    }
}
```

## 📊 Interfaz Visual

```
TELEOPERACION - FLECHAS
Estado: CONECTADO
Vx (Forward): +0.50
Vy (Turn):    -0.30
Modo: NORMAL

                    [===---] Vx:+0.50
                    [---===] Vy:-0.30

CONTROLES:
↑ W = Adelante | ↓ S = Atras | ← A = Izq | → D = Der
Mantén presionado para acelerar | Q = Salir
```

## 🔍 Troubleshooting

### El robot no se mueve
1. Verifica la conexión WebRTC en consola
2. Asegúrate de que el Go2 esté en modo NORMAL (no en DAMP)
3. Aumenta `SPEED_INCREMENT` (ej: 0.3)

### Las flechas no funcionan
- En algunos sistemas, `waitKeyEx()` no reconoce las flechas
- **Solución**: Usa WASD en su lugar (ya implementado)

### Control muy sensible
- Reduce `SPEED_INCREMENT` a 0.05
- Reduce `MAX_SPEED_X/Y`

### Control muy lento
- Aumenta `SPEED_INCREMENT` a 0.2
- Aumenta `MAX_SPEED_X/Y` a 1.2

### Latencia alta
- Reduce el timeout de `cv2.waitKeyEx(50)` a 30
- Aumenta `send_move_commands()` a 15 Hz (cambiar `await asyncio.sleep(0.1)` a 0.067)

## 📚 Relación con Otros Scripts

### vs `seguimiento_autonomo.py`
- **Teleop**: Control manual con teclado
- **Seguimiento**: Seguimiento autónomo de persona

### vs `sportmode.py`
- **Teleop**: Control continuo y fluido
- **Sportmode**: Comandos discretos (Hello, BackFlip, etc)

## 💡 Tips

1. **Movimiento coordinado**: Presiona flecha arriba + derecha para avanzar y girar simultáneamente
2. **Control fino**: Usa SPEED_INCREMENT pequeño (0.05) para ajustes precisos
3. **Pruebas**: Comienza con velocidades bajas (MAX_SPEED_X = 0.3)
4. **Emergencia**: Presiona 'Q' para salir inmediatamente

## 🎯 Mejoras Futuras

- [ ] Joystick/gamepad en lugar de teclado
- [ ] Grabar secuencia de movimientos
- [ ] Patrón de patrulla automática
- [ ] Control gestos con cámara
- [ ] Velocidad variable con scroll del mouse

---
**Versión**: 1.0  
**Fecha**: 2026-03  
**Plataforma**: Go2 LocalAP/LocalSTA/Remote
