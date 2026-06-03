import asyncio
import logging
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

logging.basicConfig(level=logging.INFO)

async def main():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip="192.168.0.114",  # poné la IP real del robot
    )

    print("Conectando...")
    await conn.connect()
    print("CONNECT OK")

    # Espera corta para ver si queda estable
    await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())