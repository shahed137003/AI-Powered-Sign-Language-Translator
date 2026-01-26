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

from config.database import init_db
from routes.users import router as users_router
from routes.password_reset import router as password_router
from routes.contact import router as contact_router
from routes.sign_to_text import router as sign_to_text_router
# from routes.translate import router as translate_router

app = FastAPI(title="AI Powered Sign Language Translator")

# Add CORS middleware - CRITICAL for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
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
app.include_router(sign_to_text_router)
# app.include_router(translate_router)
@app.get("/")
def root():
    return {"message":"Backend is running "} 