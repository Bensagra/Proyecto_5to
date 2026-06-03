import asyncio
import json
import math
import logging
from dataclasses import dataclass
from typing import Optional

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.FATAL)


@dataclass
class Go2Config:
    connection_mode: str = "LOCAL_STA_IP"   # LOCAL_STA_IP | LOCAL_STA_SN | LOCAL_AP | REMOTE
    robot_ip: str = "192.168.8.181"
    robot_sn: str = "B42D2000XXXXXXXX"
    username: str = "email@gmail.com"
    password: str = "tu_password"
    linear_speed_mps: float = 0.5
    angular_speed_rad_s: float = 0.8
    control_interval: float = 0.1
    mode_switch_wait_s: float = 3.0
    post_stop_wait_s: float = 0.2


@dataclass
class MotionCommand:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Go2Controller:
    def __init__(self, config: Optional[Go2Config] = None):
        self.config = config or Go2Config()
        self.conn: Optional[UnitreeWebRTCConnection] = None

    def _build_connection(self) -> UnitreeWebRTCConnection:
        mode = self.config.connection_mode

        if mode == "LOCAL_STA_IP":
            return UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=self.config.robot_ip,
            )

        if mode == "LOCAL_STA_SN":
            return UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                serialNumber=self.config.robot_sn,
            )

        if mode == "LOCAL_AP":
            return UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalAP
            )

        if mode == "REMOTE":
            return UnitreeWebRTCConnection(
                WebRTCConnectionMethod.Remote,
                serialNumber=self.config.robot_sn,
                username=self.config.username,
                password=self.config.password,
            )

        raise ValueError(f"Modo de conexión inválido: {mode}")

    async def connect(self):
        self.conn = self._build_connection()
        await self.conn.connect()
        await self.ensure_normal_mode()
        await self.stop()

    async def disconnect(self):
        if not self.conn:
            return

        try:
            await self.stop()
        except Exception:
            pass

        try:
            await self.conn.disconnect()
        except Exception:
            pass

        self.conn = None

    async def ensure_normal_mode(self):
        if not self.conn:
            raise RuntimeError("El robot no está conectado.")

        response = await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001},
        )

        current_mode = None
        try:
            if response["data"]["header"]["status"]["code"] == 0:
                data = json.loads(response["data"]["data"])
                current_mode = data.get("name")
        except Exception:
            current_mode = None

        if current_mode != "normal":
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {
                    "api_id": 1002,
                    "parameter": {"name": "normal"},
                },
            )
            await asyncio.sleep(self.config.mode_switch_wait_s)

        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["StandUp"]},
            )
            await asyncio.sleep(1.0)
        except Exception:
            pass

    async def send_move(self, cmd: MotionCommand):
        if not self.conn:
            raise RuntimeError("El robot no está conectado.")

        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Move"],
                "parameter": {
                    "x": cmd.x,
                    "y": cmd.y,
                    "z": cmd.z,
                },
            },
        )

    async def stop(self):
        if not self.conn:
            return

        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            )
        except Exception:
            pass

        try:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["StopMove"]},
            )
        except Exception:
            pass

        await asyncio.sleep(self.config.post_stop_wait_s)

    async def execute_timed_motion(
        self,
        cmd: MotionCommand,
        duration_s: float,
    ):
        if not self.conn:
            raise RuntimeError("El robot no está conectado.")

        if duration_s <= 0:
            raise ValueError("La duración debe ser mayor a 0.")

        elapsed = 0.0
        try:
            while elapsed < duration_s:
                await self.send_move(cmd)
                await asyncio.sleep(self.config.control_interval)
                elapsed += self.config.control_interval
        finally:
            await self.stop()

    async def move_forward(self, meters: float):
        if meters <= 0:
            raise ValueError("Los metros deben ser mayores a 0.")

        duration = meters / abs(self.config.linear_speed_mps)
        cmd = MotionCommand(x=abs(self.config.linear_speed_mps), y=0.0, z=0.0)
        await self.execute_timed_motion(cmd, duration)

    async def move_backward(self, meters: float):
        if meters <= 0:
            raise ValueError("Los metros deben ser mayores a 0.")

        duration = meters / abs(self.config.linear_speed_mps)
        cmd = MotionCommand(x=-abs(self.config.linear_speed_mps), y=0.0, z=0.0)
        await self.execute_timed_motion(cmd, duration)

    async def turn_right(self, degrees: float):
        if degrees <= 0:
            raise ValueError("Los grados deben ser mayores a 0.")

        radians = math.radians(degrees)
        duration = radians / abs(self.config.angular_speed_rad_s)
        cmd = MotionCommand(x=0.0, y=0.0, z=-abs(self.config.angular_speed_rad_s))
        await self.execute_timed_motion(cmd, duration)

    async def turn_left(self, degrees: float):
        if degrees <= 0:
            raise ValueError("Los grados deben ser mayores a 0.")

        radians = math.radians(degrees)
        duration = radians / abs(self.config.angular_speed_rad_s)
        cmd = MotionCommand(x=0.0, y=0.0, z=abs(self.config.angular_speed_rad_s))
        await self.execute_timed_motion(cmd, duration)

    async def execute_action(self, action: str, value: float):
        action = action.strip().lower()

        if action == "adelante":
            await self.move_forward(value)
        elif action in ("atras", "atrás"):
            await self.move_backward(value)
        elif action == "derecha":
            await self.turn_right(value)
        elif action == "izquierda":
            await self.turn_left(value)
        else:
            raise ValueError(
                "Acción inválida. Usá: adelante, atras, derecha, izquierda."
            )