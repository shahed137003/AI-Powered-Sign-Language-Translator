from fastapi import FastAPI
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
import models.user      
import models.password_reset
import models.contact
import models.message

from config.database import init_db
from routes.users import router as users_router
from routes.password_reset import router as password_router
from routes.contact import router as contact_router
from routes.sign_to_text import router as sign_to_text_router
from routes.chat_ws import router as chat_ws_router
from routes.chat_history import router as chat_history_router
from routes.video_call import router as video_call_router
from routes.chatbot import router as chatbot_router
# from routes.translate import router as translate_router

app = FastAPI(title="AI Powered Sign Language Translator")

# Add CORS middleware - CRITICAL for frontend-backend communication
# allow_origins must list specific origins when allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "capacitor://localhost",       # Capacitor Android
        "ionic://localhost",           # Ionic/Capacitor iOS
        "https://localhost",
        "http://localhost",
        # Allow all local network IPs (192.168.x.x range)
        # Note: For production, restrict this to your actual domains
    ],
    allow_origin_regex=r"http://192\.168\.\d+\.\d+(:\d+)?",  # Allow all 192.168.x.x IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
init_db()

# Include routers
app.include_router(users_router)
app.include_router(password_router)
app.include_router(contact_router)
# app.include_router(sign_to_text_router)
app.include_router(chat_ws_router)
app.include_router(chat_history_router)
app.include_router(video_call_router)
app.include_router(chatbot_router)
# app.include_router(translate_router)
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "backend"}

@app.get("/")
def root():
    return {"message":"Backend is running "} 