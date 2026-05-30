from  fastapi import APIRouter, WebSocket
from controllers.translate_ws import translate_socket

router = APIRouter(prefix="/translate", tags=["translate"])

@router.websocket("/ws")
async def translate_ws(websocket: WebSocket):
    await translate_socket(websocket)