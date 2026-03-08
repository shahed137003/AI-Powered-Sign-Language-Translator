from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import jwt, JWTError
from config.settings import settings
from sqlalchemy.orm import Session
from config.database import SessionLocal
from models.user import User
from controllers.chat_controller import ChatController
from collections import defaultdict
router = APIRouter()
active_connections = defaultdict(lambda: None)  # username -> WebSocket

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):

    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    token = token

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
    except JWTError:
        await websocket.close(code=1008)
        return

    db: Session = SessionLocal()
    user = db.query(User).filter(User.email == email).first()

    if not user:
        await websocket.close()
        return

    await websocket.accept()

    active_connections[user.username] = websocket

    try:
        while True:
            data = await websocket.receive_json()

            receiver_username = data["to"]
            content = data["message"]

            receiver = ChatController.get_user_by_username(db, receiver_username)

            if not receiver:
                await websocket.send_json({"error": "Receiver not found"})
                continue

            msg = ChatController.save_message(
                db, user.id, receiver.id, content
            )

            payload = {
                "id": msg.id,
                "sender_id": user.id,
                "receiver_id": receiver.id,
                "content": content,
                "created_at": str(msg.created_at)
            }

            # send to receiver
            if receiver_username in active_connections:
                await active_connections[receiver_username].send_json(payload)

            # send back to sender
            await websocket.send_json(payload)

    except WebSocketDisconnect:
        del active_connections[user.username]
        db.close()