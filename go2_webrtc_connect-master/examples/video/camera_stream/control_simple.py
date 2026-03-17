"""
CONTROL SIMPLE DEL GO2 CON FLECHAS
Sin cámara, sin procesamiento de video
Solo movimiento manual con teclado
"""

import asyncio
import logging
import sys
import time
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.FATAL)

# =========================
# CONFIG
# =========================
MAX_SPEED_X = 1.0
MAX_SPEED_Y = 0.5
SPEED_INCREMENT = 0.1


# =========================
# ROBOT STATE
# =========================
class RobotControl:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.connected = False
        self.running = True

    def update_from_input(self, direction):
        """Actualiza velocidades basado en dirección"""
        if direction == "up" or direction == "w":
            self.vx = min(self.vx + SPEED_INCREMENT, MAX_SPEED_X)
            print(f"↑ Adelante | Vx: {self.vx:.2f}")
        
        elif direction == "down" or direction == "s":
            self.vx = max(self.vx - SPEED_INCREMENT, -MAX_SPEED_X)
            print(f"↓ Atrás | Vx: {self.vx:.2f}")
        
        elif direction == "left" or direction == "a":
            self.vy = min(self.vy + SPEED_INCREMENT, MAX_SPEED_Y)
            print(f"← Izquierda | Vy: {self.vy:.2f}")
        
        elif direction == "right" or direction == "d":
            self.vy = max(self.vy - SPEED_INCREMENT, -MAX_SPEED_Y)
            print(f"→ Derecha | Vy: {self.vy:.2f}")
        
        elif direction == "space":
            self.vx = 0.0
            self.vy = 0.0
            print("⏹ STOP | Vx: 0.00, Vy: 0.00")

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0


# =========================
# INPUT HANDLER
# =========================
class InputHandler:
    """Maneja input de teclado (puede ser bloqueante)"""
    
    @staticmethod
    def get_input():
        """Obtiene input del usuario"""
        try:
            command = input("Comando (↑↓←→ o wasd, space=stop, q=salir): ").lower().strip()
            
            if command in ["q", "quit", "exit"]:
                return "quit"
            elif command in ["up", "w"]:
                return "up"
            elif command in ["down", "s"]:
                return "down"
            elif command in ["left", "a"]:
                return "left"
            elif command in ["right", "d"]:
                return "right"
            elif command == "space":
                return "space"
            else:
                return None
        except (EOFError, KeyboardInterrupt):
            return "quit"


# =========================
# MAIN
# =========================
async def main():
    # Conexión
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    
    robot = RobotControl()
    
    # Esperar conexión
    print("\n" + "="*60)
    print("🎮 CONTROL GO2 - FLECHAS (SIN CÁMARA)")
    print("="*60)
    print("\n🔌 Conectando al Go2...")
    
    try:
        await conn.connect()
        robot.connected = True
        print("✓ Conectado al Go2\n")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return
    
    # Esperar datachannel
    print("⏳ Esperando datachannel...")
    max_wait = 20
    start = time.time()
    
    while time.time() - start < max_wait:
        try:
            if hasattr(conn, 'datachannel') and conn.datachannel:
                if hasattr(conn.datachannel, 'pub_sub') and conn.datachannel.pub_sub:
                    print("✓ Datachannel listo\n")
                    break
        except:
            pass
        await asyncio.sleep(0.5)
    else:
        print("❌ Timeout esperando datachannel")
        return
    
    # Mostrar ayuda
    print("="*60)
    print("CONTROLES:")
    print("  ↑ W = Adelante     ↓ S = Atrás")
    print("  ← A = Izquierda    → D = Derecha")
    print("  ESPACIO = Detener")
    print("  Q = Salir")
    print("="*60)
    print("\nEscribe comandos (ejemplo: w, a, d, s, space, q):\n")
    
    # Loop de control
    while robot.running:
        try:
            # Obtener input (bloqueante)
            command = input(">> ").lower().strip()
            
            if not command:
                continue
            
            if command == "q":
                print("\n🛑 Deteniendo...")
                robot.reset()
                break
            
            # Actualizar velocidad
            robot.update_from_input(command)
            
            # Enviar comando
            try:
                move_cmd = {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": robot.vx,
                        "y": robot.vy,
                        "z": 0
                    }
                }
                
                response = await conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    move_cmd,
                    timeout=5
                )
                
            except asyncio.TimeoutError:
                print("⚠️  Timeout")
            except Exception as e:
                print(f"⚠️  Error: {e}")
        
        except KeyboardInterrupt:
            print("\n\n⏹ Interrupción por usuario")
            robot.reset()
            break
        except EOFError:
            print("\n\n⏹ Fin de input")
            robot.reset()
            break
    
    # Detener
    print("✓ Control terminado")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
