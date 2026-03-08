from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sign_to_text_service import SignToTextService
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import torch
import sys
from fastapi import WebSocket

# --- IMPORT YOUR PYTORCH TCN & UTILS ---
from asl_utils import config, preprocess_sequence_global, hybrid_frame_strategy
from train_model import TCN, FEATURE_DIM
app = FastAPI()

# CORS for HTTP requests (not WS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DEVICE = "cpu"
MODEL_PATH = "tcn_best_cpu.pth"
LABEL_PATH = "label_encoder.npy"

try:
    LABELS = np.load(LABEL_PATH, allow_pickle=True)
    num_classes = len(LABELS)
    model = TCN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ PyTorch TCN Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure tcn_best_cpu.pth and label_encoder.npy are in the ai_service folder.")
    sys.exit(1)


@app.websocket("/ws/sign-to-text")
async def sign_to_text_ws(websocket: WebSocket):
    await websocket.accept()
    service = SignToTextService(model, LABELS, DEVICE)
    print("✅ AI WS connected")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)  # now expecting JSON from frontend

            if data.get("type") == "keypoints":
                result = service.process_keypoints(data["data"])
                await websocket.send_json(result)
            elif data.get("end"):
                result = service.predict_sequence()
                await websocket.send_json(result)

    except WebSocketDisconnect:
        print("🔴 Frontend disconnected from AI WS")
    except Exception as e:
        print("💥 AI WS error:", e)




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