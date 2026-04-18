from fastapi import APIRouter, WebSocket
from controllers.translate_ws import sign_to_text

router = APIRouter()

@router.websocket("/ws/translate/sign-to-text")
async def ws_translate(websocket: WebSocket):
    """
    Frontend WebSocket entry point.
    """
    await sign_to_text(websocket)
# @router.post("/sign-to-voice")
# async def sign_to_voice_api(video: UploadFile = File(...)):
#     audio_path = sign_to_voice(video)
#     return {"audio_url": audio_path}

# @router.post("/text-to-sign")
# async def text_to_sign_api(text: str):
#     video_path = text_to_sign(text)
#     return {"video_url": video_path}

# @router.post("/voice-to-sign")
# async def voice_to_sign_api(audio: UploadFile = File(...)):
#     video_path = voice_to_sign(audio)
#     return {"video_url": video_path}