import asyncio
import json
import websockets
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

AI_WS_URL = "ws://127.0.0.1:8001/ws/sign-to-text"

async def sign_to_text(websocket: WebSocket):
    """
    Bridge between frontend and AI service.
    Forwards keypoints with hands_visible flag.
    """
    await websocket.accept()
    print("✅ Frontend connected to backend bridge")

    try:
        async with websockets.connect(AI_WS_URL) as ai_ws:
            print("🤖 Backend connected to AI service on port 8001")

            async def frontend_to_ai():
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    # Ensure hands_visible is included if not present
                    if data.get("type") == "keypoints" and "hands_visible" not in data:
                        # Default to True if not specified
                        data["hands_visible"] = True
                    
                    await ai_ws.send(json.dumps(data))

            async def ai_to_frontend():
                while True:
                    result = await ai_ws.recv()
                    await websocket.send_text(result)

            await asyncio.gather(frontend_to_ai(), ai_to_frontend())

    except WebSocketDisconnect:
        print("🔴 Frontend disconnected")
    except websockets.exceptions.ConnectionClosed:
        print("🔴 AI service disconnected")
    except Exception as e:
        print(f"💥 Backend WS error: {e}")
