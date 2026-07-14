import sys
import os
import logging
from api.models.gesture_models import GestureCommand
from api.models.robot_models import MoveCommand
from api.services.websocket_service import ws_manager
from api.config import SERVO_MIN, SERVO_MAX

# Add parent directory to path to import protocol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import protocol

logger = logging.getLogger(__name__)

class RobotService:
    def __init__(self):
        self._sequence = 0

    def clamp(self, value: float) -> float:
        return max(SERVO_MIN, min(SERVO_MAX, value))

    async def process_gesture(self, cmd: GestureCommand):
        # Validate and clamp
        j1 = self.clamp(cmd.joint1)
        j2 = self.clamp(cmd.joint2)
        j3 = self.clamp(cmd.joint3)
        j4 = self.clamp(cmd.joint4)
        j5 = self.clamp(cmd.joint5)
        gripper = self.clamp(cmd.gripper)
        
        self._sequence = (self._sequence + 1) % 256

        # Convert to protocol packet
        try:
            packet = protocol.build_packet(
                self._sequence,
                cmd.mode,
                j1, j2, j3, j4, j5, gripper
            )
            await ws_manager.send_packet(packet)
        except Exception as e:
            logger.error(f"Failed to build or send packet: {e}")

    async def move_robot(self, cmd: MoveCommand):
        j1 = self.clamp(cmd.joint1)
        j2 = self.clamp(cmd.joint2)
        j3 = self.clamp(cmd.joint3)
        j4 = self.clamp(cmd.joint4)
        j5 = self.clamp(cmd.joint5)
        gripper = self.clamp(cmd.gripper)
        
        self._sequence = (self._sequence + 1) % 256

        try:
            packet = protocol.build_packet(
                self._sequence,
                cmd.mode,
                j1, j2, j3, j4, j5, gripper
            )
            await ws_manager.send_packet(packet)
        except Exception as e:
            logger.error(f"Failed to build or send packet: {e}")

robot_service = RobotService()
