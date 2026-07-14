from fastapi import APIRouter
from api.models.robot_models import MoveCommand
from api.services.robot_service import robot_service

router = APIRouter(prefix="/robot", tags=["robot"])

@router.post("/move")
async def move_robot(cmd: MoveCommand):
    await robot_service.move_robot(cmd)
    return {"status": "ok"}
