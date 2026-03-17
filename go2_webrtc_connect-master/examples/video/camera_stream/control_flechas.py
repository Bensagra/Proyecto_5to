"""
CONTROL GO2 CON TECLADO EN TIEMPO REAL
Sin cámara ni procesamiento de video
Captura directa de flechas con OpenCV
"""

import asyncio
import logging
import sys
import cv2
import time
import threading
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.FATAL)

# =========================
# CONFIG
# =========================
WINDOW_NAME = "Go2 Control - Flechas / WASD"
MAX_SPEED_X = 1.0
MAX_SPEED_Y = 0.5
SPEED_INCREMENT = 0.1

# Códigos de tecla
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_W = ord('w')
KEY_A = ord('a')
KEY_S = ord('s')
KEY_D = ord('d')
KEY_SPACE = ord(' ')
KEY_Q = ord('q')

# =========================
# ROBOT STATE
# =========================
class RobotControl:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.connected = False
        self.last_command_time = time.time()

    def update_velocity(self, key):
        """Actualiza velocidades con tecla"""
        if key == KEY_UP or key == KEY_W:
            self.vx = min(self.vx + SPEED_INCREMENT, MAX_SPEED_X)
        elif key == KEY_DOWN or key == KEY_S:
            self.vx = max(self.vx - SPEED_INCREMENT, -MAX_SPEED_X)
        elif key == KEY_LEFT or key == KEY_A:
            self.vy = min(self.vy + SPEED_INCREMENT, MAX_SPEED_Y)
        elif key == KEY_RIGHT or key == KEY_D:
            self.vy = max(self.vy - SPEED_INCREMENT, -MAX_SPEED_Y)
        elif key == KEY_SPACE:
            self.vx = 0.0
            self.vy = 0.0
        
        self.last_command_time = time.time()

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0


# =========================
# MAIN
# =========================
def main():
    robot = RobotControl()
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    
    loop = None
    asyncio_thread = None

    # =========================
    # ASYNCIO FUNCTIONS
    # =========================
    def run_asyncio_loop(loop_inst):
        """Ejecuta loop asyncio en thread - Patrón robusto"""
        asyncio.set_event_loop(loop_inst)

        async def setup():
            """Conecta y configura"""
            try:
                print(f"\n🔌 Conectando al Go2...")
                await conn.connect()
                robot.connected = True
                print("✓ Conectado\n")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Error de conexión: {e}")
                robot.connected = False
                return False

            try:
                print("⏳ Esperando datachannel...")
                max_wait = 20
                start = time.time()
                while time.time() - start < max_wait:
                    try:
                        if hasattr(conn, 'datachannel') and conn.datachannel:
                            if hasattr(conn.datachannel, 'pub_sub') and conn.datachannel.pub_sub:
                                print("✓ Datachannel listo\n")
                                return True
                    except:
                        pass
                    await asyncio.sleep(0.5)
                
                print("❌ Timeout datachannel")
                return False
            except Exception as e:
                print(f"❌ Error: {e}")
                return False

        async def send_move_loop():
            """Tarea que envía comandos continuamente"""
            while True:
                try:
                    if robot.connected:
                        move_cmd = {
                            "api_id": SPORT_CMD["Move"],
                            "parameter": {
                                "x": robot.vx,
                                "y": robot.vy,
                                "z": 0
                            }
                        }

                        try:
                            response = await asyncio.wait_for(
                                conn.datachannel.pub_sub.publish_request_new(
                                    RTC_TOPIC["SPORT_MOD"],
                                    move_cmd
                                ),
                                timeout=2  # Timeout más corto
                            )
                        except asyncio.TimeoutError:
                            pass  # Ignorar, reintentaré en 100ms
                        except Exception as e:
                            logging.debug(f"Error: {e}")

                    await asyncio.sleep(0.1)  # ~10 Hz

                except Exception as e:
                    logging.debug(f"Loop error: {e}")
                    await asyncio.sleep(0.1)

        async def run():
            """Ejecuta setup y luego inicia tarea de movimiento"""
            if await setup():
                # Crear tarea de movimiento que corre en background
                loop_inst.create_task(send_move_loop())

        # Ejecutar setup y luego esperar eventos
        loop_inst.run_until_complete(run())
        loop_inst.run_forever()  # ← El patrón clave

    # =========================
    # INIT
    # =========================
    print("\n" + "="*70)
    print("🎮 CONTROL GO2 CON FLECHAS - SIN CÁMARA")
    print("="*70)

    # Crear ventana (sin video)
    blank = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, blank)
    cv2.waitKey(1)

    # Iniciar asyncio loop
    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    print("\n📋 CONTROLES:")
    print("  ↑ o W = Adelante (+Vx)")
    print("  ↓ o S = Atrás (-Vx)")
    print("  ← o A = Girar izquierda (+Vy)")
    print("  → o D = Girar derecha (-Vy)")
    print("  ESPACIO = Detener (Vx=0, Vy=0)")
    print("  Q = Salir")
    print("\n" + "="*70 + "\n")

    # =========================
    # MAIN LOOP
    # =========================
    try:
        while True:
            # Capturar tecla
            key = cv2.waitKeyEx(50)

            if key == KEY_Q or key == ord('q'):
                print("\n❌ Saliendo...")
                robot.reset()
                break

            elif key in [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
                        KEY_W, KEY_A, KEY_S, KEY_D, KEY_SPACE]:
                robot.update_velocity(key)

                # Mostrar estado
                status = f"Vx: {robot.vx:+.2f} | Vy: {robot.vy:+.2f}"
                print(f"  ⚡ {status}")

                # Actualizar ventana
                frame = blank.copy()
                cv2.putText(frame, "GO2 CONTROL", (130, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, status, (40, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "Presiona Q para salir", (40, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow(WINDOW_NAME, frame)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⏹ Interrupción")
        robot.reset()

    finally:
        print("\n🧹 Limpiando...")
        robot.reset()
        time.sleep(0.2)
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)
        print("✓ Desconectado\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error Fatal: {e}")
        sys.exit(1)
