from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import urllib.request
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat_with_bot(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Check if we can find it in the ai_service env
        ai_env_path = Path(__file__).parent.parent.parent / "ai_service" / ".env"
        if ai_env_path.exists():
            load_dotenv(dotenv_path=ai_env_path)
            api_key = os.environ.get("GROQ_API_KEY")
            
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the backend server.")
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are LinguaSign's AI sign language assistant. You help users understand sign language (e.g. ASL, BSL), how to use this translation app, and answer their general questions. Keep your answers brief, clear, and friendly."
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            reply = res_data["choices"][0]["message"]["content"]
            return {"reply": reply}
    except Exception as e:
        print(f"Chatbot Groq request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with AI service: {str(e)}")