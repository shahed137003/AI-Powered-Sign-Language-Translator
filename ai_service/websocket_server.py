from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sign_to_text_service import SignToTextService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for HTTP requests (not WS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = SignToTextService()
@app.websocket("/ws/sign-to-text")
async def sign_to_text_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ AI WS connected")

    try:
        while True:
            frame = await websocket.receive_text()
            print("📩 Frame received from frontend:", frame[:50])
            result = service.process_frame(frame)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("🔴 Frontend disconnected from AI WS")




# from fastapi import FastAPI, WebSocket , WebSocketDisconnect
# from sign_to_text_service import SignToTextService
# from fastapi.middleware.cors import CORSMiddleware
# app = FastAPI()


# # CORS (for local testing)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.websocket("/ws/sign-to-text")
# async def sign_to_text_ws(websocket: WebSocket):
#     await websocket.accept(headers=[("Access-Control-Allow-Origin", "*")])
#     service = SignToTextService()

#     print("✅ AI WebSocket connection established")

#     try:
#         while True:
#             # Receive frame (base64) from backend
#             frame = await websocket.receive_text()

#             result = service.process_frame(frame)

#             if result.get("progress", 0) < 100:
#                 response = {
#                     "text": result["text"],
#                     "confidence": float(result.get("confidence", 0))
#                 }

#                 print(
#                     f"🧠 AI Prediction: {response['text']} "
#                     f"(confidence: {response['confidence']:.2f})"
#                 )

#                 await websocket.send_json(response)

#             else:
                
#                 await websocket.send_json(result)

#     except WebSocketDisconnect:
#         print("🔴 AI Client disconnected")

#     except Exception as e:
#         print("💥 AI WS error:", e)
