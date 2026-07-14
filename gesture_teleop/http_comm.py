import requests
from transport import TransportController, TeleopCommand
import threading

class HttpController(TransportController):
    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}/gesture"
        self._connected = False
        self._last_error = ""

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str:
        return self._last_error

    def connect(self) -> None:
        try:
            res = requests.get(f"http://{self.url.split('/gesture')[0]}/health", timeout=2.0)
            if res.status_code == 200:
                self._connected = True
                self._last_error = ""
            else:
                self._last_error = f"HTTP {res.status_code}"
        except Exception as e:
            self._connected = False
            self._last_error = str(e)

    def send(self, command: TeleopCommand, force: bool = False) -> bool:
        if not force and not self._connected:
            return False

        payload = {
            "joint1": command.base_deg,
            "joint2": command.lower_deg,
            "joint3": command.middle_deg,
            "joint4": command.upper_deg,
            "joint5": command.wrist_deg,
            "gripper": command.gripper_deg,
            "mode": command.mode
        }

        try:
            res = requests.post(self.url, json=payload, timeout=0.5)
            if res.status_code == 200:
                self._connected = True
                self._last_error = ""
                return True
            else:
                self._connected = False
                self._last_error = f"HTTP Error: {res.status_code}"
                return False
        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            return False

    def close(self) -> None:
        self._connected = False
