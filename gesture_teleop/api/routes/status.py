from fastapi import APIRouter
from api.services.websocket_service import ws_manager

router = APIRouter(prefix="/robot", tags=["status"])

@router.get("/status")
async def get_status():
    return ws_manager.status
