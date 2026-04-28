import os
import sys
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add InferenceLLM to path
sys.path.append(str(Path(__file__).parent / "InferenceLLM"))
from core.pipeline import ASLPipeline
from core.llm import LLMTranslator

# ---------- Environment ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not set. Run: $env:GROQ_API_KEY='your-key'")
    sys.exit(1)

# ---------- FastAPI app ----------
app = FastAPI(title="AI Sign Language Service (LLM Pipeline)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/sign-to-text")
async def sign_to_text_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected – creating new ASL pipeline")
    llm = LLMTranslator(api_key=GROQ_API_KEY)
    pipeline = ASLPipeline(llm)

    # --- DEDUPLICATION: remember last sent word/sentence ---
    last_sent_word = None
    last_sent_sentence = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("type") == "keypoints":
                hands_visible = data.get("hands_visible", True)
                keypoints = np.array(data["data"])
                result = pipeline.process_keypoints(keypoints, hands_visible)

                if result is None:
                    # Still collecting – send progress
                    await websocket.send_json({
                        "status": "collecting",
                        "frames_collected": len(pipeline.buffer)
                    })
                else:
                    pred, conf, words, sentence = result
                    # Send word only if confidence high and it's a new word
                    if pred != "Waiting..." and conf > 0 and pred != last_sent_word:
                        last_sent_word = pred
                        await websocket.send_json({
                            "status": "success",
                            "text": pred,
                            "confidence": conf / 100.0,
                            "sentence_buffer": " ".join(words),
                            "english_sentence": sentence
                        })
                    # Send sentence only if changed
                    if sentence and sentence != "Waiting for more words..." and sentence != last_sent_sentence:
                        last_sent_sentence = sentence
                        await websocket.send_json({
                            "status": "sentence",
                            "sentence": sentence
                        })

            elif data.get("type") == "reset":
                pipeline.reset()
                # Reset the deduplication state
                last_sent_word = None
                last_sent_sentence = None
                await websocket.send_json({"status": "reset", "message": "Buffers cleared"})

            elif data.get("type") == "translate":
                # Manual LLM translation (like pressing 't')
                gloss = " ".join(pipeline.sentence_buffer)
                if gloss.strip():
                    print(f"📤 Manual translation: '{gloss}'")
                    complete, english = llm.translate(gloss)
                    if complete:
                        pipeline.english_sentence = english
                        last_sent_sentence = english
                        await websocket.send_json({
                            "status": "sentence",
                            "sentence": english
                        })
                    else:
                        await websocket.send_json({
                            "status": "error",
                            "text": "LLM translation failed"
                        })
                else:
                    await websocket.send_json({
                        "status": "error",
                        "text": "No words to translate"
                    })

    except WebSocketDisconnect:
        print("🔴 Client disconnected")
    except Exception as e:
        print(f"💥 WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 AI Service (LLM Pipeline) running on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)