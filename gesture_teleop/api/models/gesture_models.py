from pydantic import BaseModel, Field

class GestureCommand(BaseModel):
    joint1: float = Field(..., description="Base yaw angle (degrees)")
    joint2: float = Field(..., description="Lower pitch angle (degrees)")
    joint3: float = Field(..., description="Middle pitch angle (degrees)")
    joint4: float = Field(..., description="Upper pitch angle (degrees)")
    joint5: float = Field(..., description="Wrist roll angle (degrees)")
    gripper: float = Field(..., description="Gripper angle (degrees)")
    mode: str = Field("T", description="Command mode (e.g. 'T' for teleop)")
