import argparse
import asyncio
import queue
import threading
import time

import cv2
import numpy as np
from aiortc import MediaStreamTrack

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conecta a la camara del Go2 e imprime valores basicos del frame"
    )
    parser.add_argument("--method", choices=["ap", "sta"], default="ap", help="Metodo de conexion")
    parser.add_argument("--ip", default="", help="IP del robot para modo STA")
    parser.add_argument("--serial", default="", help="Serial del robot para modo STA")
    parser.add_argument("--print-every", type=float, default=0.5, help="Segundos entre prints")
    parser.add_argument("--show", action="store_true", help="Mostrar ventana de video")
    parser.add_argument("--startup-timeout", type=float, default=25.0, help="Timeout de arranque")
    return parser.parse_args()


def build_connection(args):
    if args.method == "ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    kwargs = {}
    if args.ip:
        kwargs["ip"] = args.ip
    if args.serial:
        kwargs["serialNumber"] = args.serial

    if not kwargs:
        raise ValueError("Para --method sta debes pasar --ip o --serial")

    return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, **kwargs)


def main():
    args = parse_args()
    conn = build_connection(args)

    frame_queue = queue.Queue(maxsize=2)
    loop = asyncio.new_event_loop()

    running = True
    connected_event = threading.Event()
    failed_event = threading.Event()
    failure_message = {"text": ""}

    async def recv_camera_stream(track: MediaStreamTrack):
        while running:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put(img)

    def run_asyncio_loop():
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.add_track_callback(recv_camera_stream)
                conn.video.switchVideoChannel(True)
                connected_event.set()
            except BaseException as exc:
                failure_message["text"] = f"{type(exc).__name__}: {exc}"
                failed_event.set()

        loop.run_until_complete(setup())
        if connected_event.is_set():
            loop.run_forever()

    worker = threading.Thread(target=run_asyncio_loop, daemon=True)
    worker.start()

    print("[INFO] Conectando a la camara...")
    start_wait = time.time()
    while not connected_event.is_set() and not failed_event.is_set():
        if time.time() - start_wait > args.startup_timeout:
            print(f"[ERROR] No conecto en {args.startup_timeout:.1f}s")
            running = False
            break
        time.sleep(0.05)

    if failed_event.is_set():
        print(f"[ERROR] Fallo de conexion: {failure_message['text']}")
        running = False

    last_print = 0.0
    frame_count = 0
    t0 = time.time()

    try:
        while running and connected_event.is_set():
            try:
                img = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame_count += 1
            h, w = img.shape[:2]

            center_bgr = img[h // 2, w // 2].tolist()
            mean_bgr = img.mean(axis=(0, 1))

            now = time.time()
            if now - last_print >= max(0.05, args.print_every):
                fps = frame_count / max(now - t0, 1e-6)
                print(
                    "Frame:",
                    f"{w}x{h}",
                    "| dtype:",
                    img.dtype,
                    "| center_bgr:",
                    center_bgr,
                    "| mean_bgr:",
                    np.round(mean_bgr, 2).tolist(),
                    "| fps:",
                    f"{fps:.1f}",
                )
                last_print = now

            if args.show:
                cv2.imshow("Go2 Camara", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        running = False

        try:
            future = asyncio.run_coroutine_threadsafe(conn.disconnect(), loop)
            future.result(timeout=3)
        except Exception:
            pass

        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

        worker.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
