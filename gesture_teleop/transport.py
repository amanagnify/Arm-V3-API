from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import protocol

if TYPE_CHECKING:
    from config import AppConfig


@dataclass
class TeleopCommand:
    """One control frame. ALL six joint fields are ABSOLUTE servo angles in
    [0,180] degrees — including the gripper, which the host converts from its
    [0,1] open fraction (open=166.5, closed=36) before building the command.

    Field names keep their historical spelling so downstream code stays stable:
        base_deg   -> base yaw (absolute, integrated on the Python side)
        lower_deg  -> p1 (lower pitch)
        middle_deg -> p2 (middle pitch, host direction follows `p2_invert`)
        upper_deg  -> p3 (upper pitch)
        wrist_deg  -> roll
        gripper_deg -> gripper servo angle
    """

    mode: str
    base_deg: float
    lower_deg: float
    middle_deg: float
    upper_deg: float
    wrist_deg: float
    gripper_deg: float
    sequence: int = 0

    def to_packet(self) -> bytes:
        return protocol.build_packet(
            self.sequence,
            self.mode,
            self.base_deg,
            self.lower_deg,
            self.middle_deg,
            self.upper_deg,
            self.wrist_deg,
            self.gripper_deg,
        )


class TransportController(ABC):
    @property
    @abstractmethod
    def connected(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def last_error(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send(self, command: TeleopCommand, force: bool = False) -> bool:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def build_transport(config: AppConfig) -> TransportController:
    kind = config.transport.kind.lower()

    if kind == "http":
        from http_comm import HttpController
        # Default API address fallback if not specified
        host = getattr(config, 'http_host', '127.0.0.1')
        port = getattr(config, 'http_port', 8000)
        return HttpController(host, port)

    if kind in ("serial", "udp", "tcp", "websocket"):
        raise ValueError(
            f"Transport '{kind}' is deprecated or parked. "
            "Use 'http' to communicate via the FastAPI middleware."
        )

    raise ValueError(f"Unsupported transport kind: {config.transport.kind}")
