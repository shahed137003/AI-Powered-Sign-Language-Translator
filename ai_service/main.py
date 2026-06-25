import os
import sys
import json
import asyncio
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import torch
torch.set_num_threads(4)  # Optimize CPU usage
torch.backends.cudnn.benchmark = True  # If using GPU
# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
INFERENCE = BASE / "InferenceLLM"
PREPROC = INFERENCE / "Preprocessing_Landmarks"

sys.path.insert(0, str(INFERENCE))
sys.path.insert(0, str(PREPROC))

# ── Imports ───────────────────────────────────────────────────────────────────
from core.model import TCN
from core.pipeline import ASLPipeline
from core.llm import LLMTranslator
from core.config import MIN_FRAMES, CONF_THRESHOLD
from preprocessing.constants import FEATURE_DIM

print("=" * 60)
print("AI SIGN LANGUAGE SERVICE WITH FEATURE ENGINEERING")
print("=" * 60)

# ── Validate environment ──────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("GROQ_API_KEY not set. Add it to your .env file.")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
MODEL_PATH = INFERENCE / "models_features" / "best_model.pth"
LABEL_PATH = INFERENCE / "models_features" / "label_encoder.npy"

print(f"\nModel : {MODEL_PATH}")
print(f"Labels: {LABEL_PATH}")
print(f"Feature dim: {FEATURE_DIM}")

# Load labels
try:
    LABELS_DATA = np.load(LABEL_PATH, allow_pickle=True)
    if isinstance(LABELS_DATA, np.ndarray) and LABELS_DATA.ndim == 0:
        LABELS_DATA = LABELS_DATA.item()
    if isinstance(LABELS_DATA, dict):
        sorted_classes = sorted(LABELS_DATA.keys(), key=lambda k: LABELS_DATA[k])
        LABELS = np.array(sorted_classes)
    else:
        LABELS = np.array(LABELS_DATA).astype(str)
    num_classes = len(LABELS)
    print(f"Loaded {num_classes} classes. First 10: {LABELS[:10]}")
except Exception as e:
    print(f"Error loading labels: {e}")
    sys.exit(1)

# Model expects 928 features (after feature engineering)
MODEL_INPUT_DIM = 928

# Build model
from core.model import TCN as TCNModel
model = TCNModel(
    input_dim=MODEL_INPUT_DIM,
    num_classes=num_classes,
    ch=128,
    layers=4,
    drop=0.3,
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("Model loaded.\n")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Sign Language Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/sign-to-text")
async def sign_to_text_ws(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    llm = LLMTranslator(api_key=GROQ_API_KEY)
    pipeline = ASLPipeline(model, LABELS, llm)  # ← FIXED: Pass model and labels

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("type") == "keypoints":
                hands_visible = data.get("hands_visible", True)
                keypoints = np.array(data["data"])
                result = pipeline.process_keypoints(keypoints, hands_visible)

                if result is None:
                    await websocket.send_json({
                        "status": "collecting",
                        "frames_collected": len(pipeline.buffer),
                    })
                else:
                    pred, conf, words, sentence = result
                    if pred and pred != "Waiting..." and conf > 0:
                        await websocket.send_json({
                            "status": "success",
                            "text": pred,
                            "confidence": conf / 100.0,
                            "low_confidence": conf <= CONF_THRESHOLD,
                            "sentence_buffer": " ".join(words),
                            "english_sentence": sentence or "",
                        })
                    elif sentence and sentence != "Waiting for more words...":
                        await websocket.send_json({
                            "status": "sentence",
                            "sentence": sentence
                        })

            elif data.get("type") == "end":
                result = pipeline.end_recording()
                if result:
                    pred, conf, words, sentence = result
                    if pred and pred != "Waiting..." and conf > 0:
                        await websocket.send_json({
                            "status": "success",
                            "text": pred,
                            "confidence": conf / 100.0,
                            "low_confidence": conf <= CONF_THRESHOLD,
                            "sentence_buffer": " ".join(words),
                            "english_sentence": sentence or "",
                        })
                    elif sentence and sentence != "Waiting for more words...":
                        await websocket.send_json({
                            "status": "sentence",
                            "sentence": sentence
                        })

            elif data.get("type") == "undo":
                pipeline.undo()
                if pipeline.sentence_buffer:
                    gloss_str = " ".join(pipeline.sentence_buffer)
                    _, sentence = await asyncio.to_thread(pipeline.llm.translate, gloss_str)
                    pipeline.english_sentence = sentence
                else:
                    pipeline.english_sentence = ""
                
                pred, conf, words, sentence = pipeline.last_pred, pipeline.last_conf, pipeline.sentence_buffer, pipeline.english_sentence
                await websocket.send_json({
                    "status": "success",
                    "text": pred,
                    "confidence": conf / 100.0,
                    "low_confidence": False,
                    "sentence_buffer": " ".join(words),
                    "english_sentence": sentence or "",
                })

            elif data.get("type") == "add_word":
                word = data.get("word")
                pipeline.add_word(word)
                if pipeline.sentence_buffer:
                    gloss_str = " ".join(pipeline.sentence_buffer)
                    _, sentence = await asyncio.to_thread(pipeline.llm.translate, gloss_str)
                    pipeline.english_sentence = sentence
                else:
                    pipeline.english_sentence = ""
                
                pred, conf, words, sentence = pipeline.last_pred, pipeline.last_conf, pipeline.sentence_buffer, pipeline.english_sentence
                await websocket.send_json({
                    "status": "success",
                    "text": pred,
                    "confidence": conf / 100.0,
                    "low_confidence": False,
                    "sentence_buffer": " ".join(words),
                    "english_sentence": sentence or "",
                })

            elif data.get("type") == "translate":
                if pipeline.sentence_buffer:
                    gloss_str = " ".join(pipeline.sentence_buffer)
                    _, sentence = await asyncio.to_thread(pipeline.llm.translate, gloss_str)
                    pipeline.english_sentence = sentence
                else:
                    pipeline.english_sentence = ""
                
                await websocket.send_json({
                    "status": "sentence",
                    "sentence": pipeline.english_sentence
                })

            elif data.get("type") == "reset":
                pipeline.reset()
                await websocket.send_json({"status": "reset", "message": "Reset successful"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "raw_dim": FEATURE_DIM,
        "model_dim": MODEL_INPUT_DIM,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    import uvicorn
    print("\nStarting server on http://0.0.0.0:8001")
    print("   WebSocket: ws://localhost:8001/ws/sign-to-text")
    print("   Health: http://localhost:8001/health\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)