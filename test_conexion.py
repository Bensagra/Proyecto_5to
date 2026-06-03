#!/usr/bin/env python3
"""
Script para probar conexion con el robot Go2 y encontrar su IP
"""
import subprocess
import time
import logging
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("TestConexion")

# IPs comunes para Go2
IPS_A_PROBAR = [
    "192.168.0.114",      # ← ENCONTRAMOS ESTA
    "192.168.1.100",      # Redes típicas
    "192.168.1.101",
    "192.168.8.181",      # STA default
    "192.168.12.1",       # AP default
    "10.0.0.50",
]

def ping_ip(ip: str, timeout: int = 1) -> bool:
    """Prueba si una IP responde al ping"""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout * 1000), ip],
            capture_output=True,
            timeout=timeout + 1
        )
        return result.returncode == 0
    except Exception as e:
        return False

def test_webrtc_connection(ip: str) -> bool:
    """Prueba conexión WebRTC a una IP"""
    try:
        log.info(f"  → Probando WebRTC en {ip}...")
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
        log.info(f"  ✓ Conexión creada para {ip}")
        return True
    except Exception as e:
        log.debug(f"  ✗ Error en {ip}: {e}")
        return False

def main():
    log.info("╔════════════════════════════════════════════════════╗")
    log.info("║     TEST DE CONEXIÓN CON UNITREE GO2             ║")
    log.info("╚════════════════════════════════════════════════════╝")
    log.info("")
    
    log.info("PASO 1: Probando conectividad con PING...")
    log.info("")
    
    responsive_ips = []
    for ip in IPS_A_PROBAR:
        print(f"Ping a {ip}...", end=" ", flush=True)
        if ping_ip(ip):
            print("✓ Responde")
            responsive_ips.append(ip)
        else:
            print("✗")
    
    log.info("")
    
    if not responsive_ips:
        log.warning("⚠ Ningún IP respondió al ping")
        log.warning("Esto podría significar:")
        log.warning("  1. El robot está apagado")
        log.warning("  2. No estás en la misma red")
        log.warning("  3. Hay firewall bloqueando")
        log.warning("")
        log.info("Tips:")
        log.info("  - Ver el IP frontal del robot o en su pantalla")
        log.info("  - Verificar que estés conectado a WiFi con el nombre correcto")
        log.info("  - En AP mode: conectarse a SSID del robot (ej: Go2-XXXX)")
        return
    
    log.info(f"PASO 2: Probando {len(responsive_ips)} IPs con WebRTC...")
    log.info("")
    
    working_ips = []
    for ip in responsive_ips:
        if test_webrtc_connection(ip):
            working_ips.append(ip)
    
    log.info("")
    log.info("╔════════════════════════════════════════════════════╗")
    
    if working_ips:
        log.info("║  ✓ CONEXIÓN EXITOSA CON:                       ║")
        for ip in working_ips:
            log.info(f"║    → {ip:<45} ║")
        log.info("╚════════════════════════════════════════════════════╝")
        log.info("")
        log.info("COMANDO PARA USAR:")
        log.info(f"  python seguimiento_perro_mejorado_fixed.py --ip {working_ips[0]}")
    else:
        log.warning("║  ✗ No se pudo conectar con WebRTC               ║")
        log.warning("╚════════════════════════════════════════════════════╝")
        log.warning("")
        log.warning("Próximos pasos:")
        log.warning("  1. Verifica que el robot esté encendido")
        log.warning("  2. Intenta modo AP:")
        log.warning("     python seguimiento_perro_mejorado_fixed.py --ap")
        log.warning("  3. Prueba con IPs manuales:")
        log.warning("     python test_conexion.py  (pero primero edita IPS_A_PROBAR)")

if __name__ == "__main__":
    main()
