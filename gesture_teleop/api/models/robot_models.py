from pydantic import BaseModel
from typing import List, Optional

class RobotStatus(BaseModel):
    connected: bool
    battery: float = 0.0
    temperature: float = 0.0
    servo_angles: List[float] = []

class MoveCommand(BaseModel):
    joint1: float
    joint2: float
    joint3: float
    joint4: float
    joint5: float
    gripper: float
    mode: str = "T"
