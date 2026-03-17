"""
TELEOPERACIÓN CON FLECHAS DEL TECLADO
Controla el Go2 en tiempo real con las teclas de flecha
"""

import asyncio
import logging
import json
import sys
import cv2
import time
import threading
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack

logging.basicConfig(level=logging.FATAL)

# =========================
# CONFIG
# =========================
WINDOW_NAME = "Go2 Teleoperación - Flechas para mover"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Velocidades
MAX_SPEED_X = 1.0      # Adelante/atrás
MAX_SPEED_Y = 0.5      # Izquierda/derecha
SPEED_INCREMENT = 0.1  # Incremento por cada comando

# Códigos de tecla (waitKeyEx)
KEY_UP = 2490368       # Flecha arriba
KEY_DOWN = 2621440     # Flecha abajo
KEY_LEFT = 2424832     # Flecha izquierda
KEY_RIGHT = 2555904    # Flecha derecha
KEY_Q = ord('q')       # Salir

# Fallback para sistemas que no reconocen waitKeyEx
KEY_W = ord('w')       # W = adelante
KEY_A = ord('a')       # A = izquierda
KEY_S = ord('s')       # S = atrás
KEY_D = ord('d')       # D = derecha

# =========================
# ESTADO DEL ROBOT
# =========================
class RobotState:
    def __init__(self):
        self.vx = 0.0      # Velocidad forward/backward
        self.vy = 0.0      # Velocidad left/right
        self.last_update = time.time()
        self.connected = False
        self.mode = "NORMAL"
        
    def update_velocity(self, key):
        """Actualiza velocidad basado en tecla presionada"""
        if key == KEY_UP or key == KEY_W:
            self.vx = min(self.vx + SPEED_INCREMENT, MAX_SPEED_X)
        elif key == KEY_DOWN or key == KEY_S:
            self.vx = max(self.vx - SPEED_INCREMENT, -MAX_SPEED_X)
        elif key == KEY_LEFT or key == KEY_A:
            self.vy = min(self.vy + SPEED_INCREMENT, MAX_SPEED_Y)
        elif key == KEY_RIGHT or key == KEY_D:
            self.vy = max(self.vy - SPEED_INCREMENT, -MAX_SPEED_Y)
        
        self.last_update = time.time()
    
    def apply_deadman(self, timeout=0.5):
        """Si pasa timeout sin comando, detiene el robot (deadman timeout)"""
        if time.time() - self.last_update > timeout:
            self.vx = 0.0
            self.vy = 0.0
    
    def reset(self):
        """Reset a velocidad 0"""
        self.vx = 0.0
        self.vy = 0.0
        self.last_update = time.time()


# =========================
# MAIN
# =========================
def main():
    # Conexión WebRTC
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000P7I9GF8A")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    
    robot_state = RobotState()
    frame_queue = None
    
    # Variables para el loop asyncio
    loop = None
    asyncio_thread = None
    move_task = None
    
    # =========================
    # FUNCIONES ASYNCIO
    # =========================
    async def recv_camera_stream(track: MediaStreamTrack):
        """Recibe frames de video del robot"""
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                frame_queue.put(img)
            except Exception as e:
                logging.debug(f"Error recibiendo frame: {e}")
                break
    
    async def wait_for_datachannel(max_wait=20):
        """Espera a que el datachannel esté listo (LocalAP puede ser lento)"""
        print("⏳ Esperando a que el datachannel esté listo...")
        start = time.time()
        elapsed = 0
        while elapsed < max_wait:
            try:
                if hasattr(conn, 'datachannel') and conn.datachannel:
                    if hasattr(conn.datachannel, 'pub_sub') and conn.datachannel.pub_sub:
                        print("✓ Datachannel listo")
                        return True
            except:
                pass
            await asyncio.sleep(0.5)
            elapsed = time.time() - start
        
        print(f"❌ Datachannel NO se abrió en {max_wait}s")
        return False
    
    async def send_move_commands():
        """Envía comandos de movimiento cada ~100ms"""
        # Esperar a que datachannel esté listo antes de empezar
        if not await wait_for_datachannel(max_wait=20):
            print("❌ No se puede enviar comandos - datachannel no disponible")
            return
        
        retry_count = 0
        max_retries = 3
        
        while True:
            try:
                # Aplicar deadman timeout
                robot_state.apply_deadman(timeout=0.5)
                
                # Enviar comando de movimiento
                move_cmd = {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": robot_state.vx,
                        "y": robot_state.vy,
                        "z": 0
                    }
                }
                
                response = await conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    move_cmd,
                    timeout=5  # Timeout de 5 segundos por comando
                )
                
                retry_count = 0  # Reset retry counter on success
                await asyncio.sleep(0.1)  # ~10 Hz
                
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"⚠️  Timeout en comando (intento {retry_count}/{max_retries})")
                    await asyncio.sleep(0.2)
                else:
                    print("❌ Demasiados timeouts - reconectando...")
                    return
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logging.debug(f"Error enviando comando (intento {retry_count}/{max_retries}): {e}")
                    await asyncio.sleep(0.2)
                else:
                    print(f"❌ Error persistente: {e}")
                    return
    
    async def reconnect_loop():
        """Loop de reconexión con reintentos"""
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                print(f"\n🔌 Conectando al Go2... (intento {attempt}/{max_attempts})")
                await conn.connect()
                robot_state.connected = True
                print("✓ Conectado al Go2\n")
                
                # Esperar un bit para que se estabilice
                await asyncio.sleep(1)
                
                print("📹 Activando cámara...")
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
                print("✓ Cámara activa\n")
                
                return True  # Conexión exitosa
                
            except Exception as e:
                logging.error(f"Error en conexión (intento {attempt}): {e}")
                robot_state.connected = False
                if attempt < max_attempts:
                    print(f"⏳ Reintentando en 3s...\n")
                    await asyncio.sleep(3)
        
        print(f"❌ No se pudo conectar después de {max_attempts} intentos")
        return False
    
    def run_asyncio_loop(loop_inst):
        """Ejecuta el loop asyncio en thread separado"""
        asyncio.set_event_loop(loop_inst)
        
        async def run():
            if await reconnect_loop():
                # Intentar enviar comandos en loop
                while robot_state.connected:
                    try:
                        await send_move_commands()
                    except Exception as e:
                        print(f"❌ Error en send_move_commands: {e}")
                        print("⏳ Intentando reconectar...")
                        await asyncio.sleep(2)
                        if not await reconnect_loop():
                            break
        
        loop_inst.run_until_complete(run())
    
    # =========================
    # INTERFAZ VISUAL
    # =========================
    def draw_ui(frame, robot_state):
        """Dibuja interfaz en el frame"""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Panel de información
        cv2.rectangle(output, (10, 10), (400, 200), (30, 30, 30), -1)
        
        # Título
        cv2.putText(output, "TELEOPERACION - FLECHAS", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Estado de conexión
        status_color = (0, 255, 0) if robot_state.connected else (0, 0, 255)
        status_text = "CONECTADO" if robot_state.connected else "DESCONECTADO"
        cv2.putText(output, f"Estado: {status_text}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Velocidades
        cv2.putText(output, f"Vx (Forward): {robot_state.vx:+.2f}", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(output, f"Vy (Turn):    {robot_state.vy:+.2f}", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Modo
        cv2.putText(output, f"Modo: {robot_state.mode}", (20, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Controles en pantalla
        cv2.rectangle(output, (10, h - 180), (w - 10, h - 10), (30, 30, 30), -1)
        
        cv2.putText(output, "CONTROLES:", (20, h - 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(output, "↑ W = Adelante  |  ↓ S = Atras  |  ← A = Izq  |  → D = Der", (20, h - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(output, "Mantén presionado para acelerar  |  Q = Salir", (20, h - 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Gráfico de velocidades
        bar_width = 100
        bar_height = 20
        bar_x = w - 150
        bar_y = 50
        
        # Barra de Vx
        vx_bar = int(abs(robot_state.vx) / MAX_SPEED_X * bar_width)
        color_x = (0, 255, 0) if robot_state.vx >= 0 else (255, 0, 0)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + vx_bar, bar_y + bar_height),
                     color_x, -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (200, 200, 200), 2)
        cv2.putText(output, f"Vx:{robot_state.vx:+.1f}", (bar_x - 30, bar_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Barra de Vy
        bar_y += 50
        vy_bar = int(abs(robot_state.vy) / MAX_SPEED_Y * bar_width)
        color_y = (0, 255, 0) if robot_state.vy >= 0 else (255, 0, 0)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + vy_bar, bar_y + bar_height),
                     color_y, -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (200, 200, 200), 2)
        cv2.putText(output, f"Vy:{robot_state.vy:+.1f}", (bar_x - 30, bar_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return output
    
    # =========================
    # INICIALIZACIÓN
    # =========================
    frame_queue = __import__('queue').Queue(maxsize=5)
    
    # Crear loop asyncio en thread separado
    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()
    
    # Crear ventana
    blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, blank)
    cv2.waitKey(1)
    
    print("\n" + "="*60)
    print("🎮 TELEOPERACIÓN CON FLECHAS - Go2 WebRTC")
    print("="*60)
    print("\n📋 Controles:")
    print("  ↑ o W  → Adelante (aumenta)")
    print("  ↓ o S  → Atrás (aumenta negativamente)")
    print("  ← o A  → Girar izquierda")
    print("  → o D  → Girar derecha")
    print("  Q     → Salir")
    print("\n⚠️  El robot se detiene si no envías comandos (deadman timeout)")
    print("="*60 + "\n")
    
    # =========================
    # LOOP PRINCIPAL
    # =========================
    try:
        while True:
            # Capturar frame si está disponible
            if not frame_queue.empty():
                try:
                    frame = frame_queue.get_nowait()
                except:
                    frame = blank.copy()
            else:
                frame = blank.copy()
            
            # Dibujar UI
            output = draw_ui(frame, robot_state)
            cv2.imshow(WINDOW_NAME, output)
            
            # Capturar tecla (waitKeyEx para flechas)
            key = cv2.waitKeyEx(50)  # 50ms timeout
            
            if key != -1:
                if key == KEY_Q or key == ord('q'):
                    print("\n❌ Saliendo...")
                    break
                elif key in [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, 
                            KEY_W, KEY_A, KEY_S, KEY_D]:
                    robot_state.update_velocity(key)
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupción por usuario")
    
    finally:
        # Limpiar
        print("\n🧹 Limpiando...")
        
        # Detener robot
        robot_state.reset()
        
        # Dar tiempo para enviar último comando
        time.sleep(0.2)
        
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)
        
        print("✓ Desconectado\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
