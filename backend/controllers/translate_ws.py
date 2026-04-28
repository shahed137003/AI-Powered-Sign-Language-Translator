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
# async def sign_to_text(websocket: WebSocket):
#     """
#     Bridge between frontend and AI service.
#     Receives frames from frontend and sends to AI WS, then returns results.
#     """
#     await websocket.accept()  # Accept frontend connection
#     ai_ws = None

#     try:
#         # Connect to AI service
#         async with websockets.connect(AI_WS_URL) as ai_ws:
#             print("🔗 Connected to AI service")

#             while True:
#                 # Receive frame from frontend
#                 frame = await websocket.receive_text()

#                 # Send frame to AI service
#                 await ai_ws.send(frame)

#                 # Receive AI result
#                 try:
#                     result = await asyncio.wait_for(ai_ws.recv(), timeout=2.0)
#                 except asyncio.TimeoutError:
#                     # If AI is still processing, send intermediate status
#                     await websocket.send_json({
#                         "status": "processing"
#                     })
#                     continue

#                 # Parse result and send back to frontend
#                 try:
#                     data = json.loads(result)
#                     await websocket.send_json(data)
#                 except json.JSONDecodeError:
#                     # If not JSON, send as plain text
#                     await websocket.send_text(result)

#     except WebSocketDisconnect:
#         print("🔴 Frontend disconnected")

#     except websockets.exceptions.ConnectionClosed:
#         print("🔴 AI service disconnected")

#     except Exception as e:
#         print(f"💥 Backend error: {e}")
#         await websocket.send_json({
#             "text": "AI service unavailable",
#             "confidence": 0.0,
#             "progress": 0
#         })