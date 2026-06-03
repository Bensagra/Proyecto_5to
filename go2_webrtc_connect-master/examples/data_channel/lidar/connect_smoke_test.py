#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMOKE TEST de conexion al Go2 — sin LiDAR, sin camara, sin nada.

Sirve para diagnosticar si el WebRTC handshake termina o no, asi separamos
"el robot no se conecta" de "mi pipeline tiene un bug".

Hace tres cosas:

1) Diagnostico de red: lista interfaces activas y avisa de VPNs (utun*) que
   suelen romper aiortc al robarse el trafico UDP.
2) Probe TCP a 192.168.12.1:9991 (puerto de signaling LAN). Si esto falla,
   no estas asociado al WiFi del robot.
3) await conn.connect() con timeout de 20s. Imprime estados intermedios cada
   500ms para que veas EXACTAMENTE donde se traba (ICE? DTLS? validacion?).

Uso:
    source .venv/bin/activate
    python connect_smoke_test.py            # default LocalAP
    python connect_smoke_test.py --ip 192.168.8.181  # LocalSTA por IP
"""

import argparse
import asyncio
import socket
import subprocess
import sys
import time

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


# =========================================================
# DIAGNOSTICOS PREVIOS
# =========================================================

def list_vpn_interfaces():
    """Devuelve nombres de interfaces VPN-like activas (utun, tun, ppp, tap)."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    active = []
    for blk in out.split("\n\n"):
        if not blk.strip():
            continue
        first = blk.split(":", 1)[0].strip()
        if first.startswith(("utun", "tun", "ppp", "tap")) and "UP" in blk and "RUNNING" in blk:
            # Solo tunneles que tengan IPv4 asignada (activos de verdad)
            if "inet " in blk:
                active.append(first)
    return active


def probe_tcp(ip: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def default_route_iface(target_ip: str) -> str:
    """Que interfaz va a usar la Mac para llegar a target_ip."""
    try:
        out = subprocess.run(
            ["route", "-n", "get", target_ip],
            capture_output=True, text=True, timeout=3,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("interface:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "?"


# =========================================================
# CONNECT WATCHER
# =========================================================

class StateWatcher:
    """Imprime el estado del peer connection cada N segundos mientras connect() corre."""

    def __init__(self, conn, period: float = 0.5):
        self.conn = conn
        self.period = period
        self.running = True
        self._last_state = None

    async def run(self):
        while self.running:
            pc = getattr(self.conn, "pc", None)
            dc = getattr(self.conn, "datachannel", None)
            ch = getattr(dc, "channel", None) if dc else None
            state = (
                getattr(pc, "connectionState", "?"),
                getattr(pc, "iceConnectionState", "?"),
                getattr(pc, "signalingState", "?"),
                getattr(ch, "readyState", "?") if ch else "?",
                getattr(dc, "data_channel_opened", False) if dc else False,
            )
            if state != self._last_state:
                self._last_state = state
                print(
                    f"  [watch] peer={state[0]:14s} ice={state[1]:10s} "
                    f"signaling={state[2]:14s} channel={state[3]:8s} validated={state[4]}",
                    flush=True,
                )
            await asyncio.sleep(self.period)


# =========================================================
# MAIN
# =========================================================

async def run_test(mode: str, ip_arg: str, serial: str, timeout: float):
    if mode == "ap":
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
        target_ip = "192.168.12.1"
    elif mode == "sta":
        if ip_arg:
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip_arg)
            target_ip = ip_arg
        else:
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=serial)
            target_ip = "(via discovery)"
    else:
        raise SystemExit(f"modo no soportado: {mode}")

    print("=" * 68)
    print(f"[1/3] Diagnostico de red")
    print("=" * 68)
    vpns = list_vpn_interfaces()
    if vpns:
        print(f"  WARN  VPN/tunneles activos: {', '.join(vpns)}")
        print("        aiortc usa UDP a un puerto efimero; si tu default route va")
        print("        por la VPN, el trafico al robot no llega. Desconectala.")
    else:
        print("  OK   no detecto interfaces VPN/tunnel activas")

    if target_ip and target_ip.startswith(("192.", "10.", "172.")):
        iface = default_route_iface(target_ip)
        print(f"  ruta hacia {target_ip} -> interfaz '{iface}'")

    print()
    print("=" * 68)
    print(f"[2/3] Probe TCP a {target_ip}:9991 (signaling LAN del Go2)")
    print("=" * 68)
    if mode == "ap" or (mode == "sta" and ip_arg):
        if probe_tcp(target_ip, 9991, timeout=3.0):
            print("  OK   puerto 9991 abierto. La Mac llega al robot por red.")
        elif probe_tcp(target_ip, 8081, timeout=3.0):
            print("  OK   puerto 8081 abierto (firmware viejo, legacy signaling).")
        else:
            print(f"  FAIL no puedo abrir TCP 9991 ni 8081 en {target_ip}.")
            print(f"       Verifica:")
            print(f"        - El robot esta prendido")
            print(f"        - La Mac esta asociada al WiFi del Go2 (SSID Unitree-Go2-XXXX)")
            print(f"        - El default gateway de tu WiFi es 192.168.12.1")
            return False
    else:
        print("  (skip: modo STA con serial usa multicast discovery; ver paso 3)")

    print()
    print("=" * 68)
    print(f"[3/3] await conn.connect() con timeout {timeout:.0f}s")
    print("=" * 68)
    print("  (vas a ver los estados intermedios para ver donde se traba)")

    watcher = StateWatcher(conn, period=0.4)
    watcher_task = asyncio.create_task(watcher.run())

    t0 = time.time()
    try:
        await asyncio.wait_for(conn.connect(), timeout=timeout)
        dt = time.time() - t0
        print(f"\n  OK   conectado en {dt:.1f}s")
        ok = True
    except asyncio.TimeoutError:
        dt = time.time() - t0
        print(f"\n  FAIL connect() supero {dt:.0f}s sin completar.")
        pc = getattr(conn, "pc", None)
        peer = getattr(pc, "connectionState", "?")
        ice = getattr(pc, "iceConnectionState", "?")
        if peer != "connected":
            print("       DIAGNOSTICO: el peer WebRTC nunca cerro el handshake.")
            print("       Causas tipicas:")
            print("        1. La app movil Unitree esta conectada (cerrala).")
            print("        2. VPN activa intercepta el UDP (desconectala).")
            print("        3. Sesion vieja del robot colgada (reboot del robot).")
        else:
            print("       DIAGNOSTICO: peer connectado pero validacion data channel falla.")
            print("       Si tu firmware es Go2 >= 1.1.15 hace falta aes_128_key.")
        ok = False
    except Exception as e:
        dt = time.time() - t0
        print(f"\n  FAIL connect() tiro excepcion despues de {dt:.1f}s: {type(e).__name__}: {e}")
        ok = False
    finally:
        watcher.running = False
        try:
            await asyncio.wait_for(watcher_task, timeout=1.0)
        except Exception:
            pass

    if ok:
        try:
            print("\n  cerrando conexion...")
            await asyncio.wait_for(conn.disconnect(), timeout=3.0)
        except Exception:
            pass

    return ok


async def run_stress(mode: str, ip_arg: str, serial: str, timeout: float, n: int, gap: float):
    """Hace N connect/disconnect seguidos para ver si el connect es flaky."""
    print(f"\n=== STRESS TEST: {n} ciclos de connect/disconnect (gap {gap}s) ===\n")
    results = []
    for i in range(1, n + 1):
        print(f"--- ciclo {i}/{n} ---")
        if mode == "ap":
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
        elif ip_arg:
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip_arg)
        else:
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=serial)

        t0 = time.time()
        try:
            await asyncio.wait_for(conn.connect(), timeout=timeout)
            dt = time.time() - t0
            print(f"  OK    conectado en {dt:.1f}s")
            results.append(("OK", dt))
            try:
                await asyncio.wait_for(conn.disconnect(), timeout=3.0)
            except Exception:
                pass
        except Exception as e:
            dt = time.time() - t0
            print(f"  FAIL  {type(e).__name__} despues de {dt:.1f}s")
            results.append(("FAIL", dt))
            try:
                await asyncio.wait_for(conn.disconnect(), timeout=3.0)
            except Exception:
                pass

        if i < n:
            print(f"  esperando {gap}s...\n")
            await asyncio.sleep(gap)

    ok = sum(1 for r, _ in results if r == "OK")
    fail = sum(1 for r, _ in results if r == "FAIL")
    print(f"\n=== STRESS SUMMARY: {ok}/{n} OK, {fail}/{n} FAIL ===")
    return ok == n


def main():
    p = argparse.ArgumentParser(description="Smoke test minimo de conexion WebRTC al Go2")
    p.add_argument("--mode", choices=["ap", "sta"], default="ap")
    p.add_argument("--ip", help="IP del robot para modo STA")
    p.add_argument("--serial", help="Serial del robot para STA con discovery")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--stress", type=int, default=0,
                   help="N>0 hace N ciclos connect/disconnect seguidos para ver si es flaky")
    p.add_argument("--stress-gap", type=float, default=5.0,
                   help="Segundos a esperar entre ciclos del stress test")
    args = p.parse_args()

    if args.mode == "sta" and not (args.ip or args.serial):
        raise SystemExit("--mode sta requiere --ip o --serial")

    if args.stress > 0:
        ok = asyncio.run(run_stress(args.mode, args.ip, args.serial,
                                    args.timeout, args.stress, args.stress_gap))
    else:
        ok = asyncio.run(run_test(args.mode, args.ip, args.serial, args.timeout))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
