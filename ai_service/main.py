import asyncio
import json
import numpy as np
import torch
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import the service and model
from sign_to_text_service import SignToTextService, TCN
from preprocessing.constants import FEATURE_DIM

app = FastAPI(title="AI Sign Language Service")

# CORS for HTTP requests (not WS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# LOAD MODEL (Using main_inference paths)
# ============================================================
DEVICE = torch.device("cpu")
MODEL_PATH = "Not cleaned + preprocessing no mask/TCN_preprocessed_no_cleaning_no_mask.pth"
LABEL_PATH = "Not cleaned + preprocessing no mask/label_encoder.npy"

try:
    # Load labels
    LABELS = np.load(LABEL_PATH, allow_pickle=True)
    num_classes = len(LABELS)
    print(f"✅ Loaded {num_classes} classes from {LABEL_PATH}")
    
    # Load model
    model = TCN(input_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure the model files exist in the correct paths:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {LABEL_PATH}")
    sys.exit(1)


@app.websocket("/ws/sign-to-text")
async def sign_to_text_ws(websocket: WebSocket):
    await websocket.accept()
    service = SignToTextService(model, LABELS, DEVICE)
    print("✅ AI WebSocket connected - Auto-sensing mode active (main_inference logic)")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "keypoints":
                # Get hands_visible flag from frontend (via backend)
                hands_visible = data.get("hands_visible", True)
                result = service.add_keypoints(data["data"], hands_visible)
                
                # Send result back to backend
                if result.get("status") == "collecting":
                    await websocket.send_json({
                        "status": "collecting",
                        "frames_collected": result.get("frames_collected", 0),
                        "progress": result.get("progress", 0)
                    })
                elif result.get("status") == "success":
                    await websocket.send_json({
                        "text": result["text"],
                        "confidence": result["confidence"],
                        "status": "success"
                    })
                elif result.get("status") == "error":
                    await websocket.send_json({
                        "text": result["text"],
                        "confidence": 0,
                        "status": "error"
                    })
                    
            elif data.get("type") == "end":
                # Manual end of recording
                result = service.force_predict()
                if result.get("status") == "success":
                    await websocket.send_json({
                        "text": result["text"],
                        "confidence": result["confidence"],
                        "status": "success"
                    })
                else:
                    await websocket.send_json({
                        "text": result["text"],
                        "confidence": 0,
                        "status": "error"
                    })
                    
            elif data.get("type") == "reset":
                service.reset()
                await websocket.send_json({"status": "reset", "message": "Service reset"})

    except WebSocketDisconnect:
        print("🔴 Frontend disconnected from AI WS")
    except Exception as e:
        print(f"💥 AI WS error: {e}")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Service on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)


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