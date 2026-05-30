from fastapi import APIRouter, WebSocket
from controllers.translate_ws import sign_to_text

router = APIRouter()

@router.websocket("/ws/translate/sign-to-text")
async def ws_translate(websocket: WebSocket):
    """
    Frontend WebSocket entry point.
    """
    await sign_to_text(websocket)
