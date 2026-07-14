from fastapi import APIRouter
from api.models.gesture_models import GestureCommand
from api.services.robot_service import robot_service

router = APIRouter(prefix="/gesture", tags=["gesture"])

@router.post("")
async def process_gesture(cmd: GestureCommand):
    await robot_service.process_gesture(cmd)
    return {"status": "ok"}
