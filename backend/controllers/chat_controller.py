from sqlalchemy.orm import Session
from models.message import Message
from models.user import User

class ChatController:
    @staticmethod
    def get_user_by_username(db: Session, username: str):
        return db.query(User).filter(User.username == username).first()
    @staticmethod
    def save_message(
        db: Session, 
        sender_id: int, 
        receiver_id: int, 
        content: str
      ):
        msg = Message(
            sender_id=sender_id, receiver_id=receiver_id, content=content
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg  
    @staticmethod
    def get_chat_history(
        db: Session, 
        user1_id: int, 
        user2_id: int
    ):
        return (
            db.query(Message)
            .filter(
                ((Message.sender_id == user1_id) & (Message.receiver_id == user2_id)) |
                ((Message.sender_id == user2_id) & (Message.receiver_id == user1_id))
            )
            .order_by(Message.created_at).all()
    )