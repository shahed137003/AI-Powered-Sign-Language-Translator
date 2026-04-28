from fastapi import APIRouter , Depends, HTTPException
from sqlalchemy.orm import Session
from config.database import get_db
from controllers.chat_controller import ChatController
from schemas.message_schema import MessageOut
from services.auth import get_current_user
from models.user import User
router = APIRouter(prefix="/chat", tags=["Chat"])

@router.get("/history/{other_username}", response_model=list[MessageOut])

def chat_history(
    other_username: str, 
    db: Session = Depends(get_db), 
    current_user = Depends(get_current_user)
):
    other_user = ChatController.get_user_by_username(db, other_username)
    if not other_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    history = ChatController.get_chat_history(db, current_user.id, other_user.id)
    return history
    