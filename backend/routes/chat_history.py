from fastapi import APIRouter , Depends
from sqlalchemy.orm import Session
from config.database import get_db
from controllers.chat_controller import ChatController
from schemas.message_schema import MessageOut

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.get("/history/{user1}/{user2}", response_model=list[MessageOut])

def chat_history(user1: str, user2: str, db: Session = Depends(get_db)):
    u1 = ChatController.get_user_by_username(db, user1)
    u2 = ChatController.get_user_by_username(db, user2)

    return ChatController.get_chat_history(db, u1.id, u2.id)