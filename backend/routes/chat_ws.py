from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from config.database import get_db
from controllers.chat_controller import ChatController

router = APIRouter()
active_connections = {}

@router.websocket("/ws/chat/{username}")
async def chat_websocket(
    websocket: WebSocket, 
    username: str, 
    db: Session = Depends(get_db)
):
    await websocket.accept()
    user = ChatController.get_user_by_username(db, username)
    if not user:
        await websocket.send_json({"error": "User not found"})
        await websocket.close()
        return
    active_connections[username] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            receiver_username = data["to"]
            content = data["message"]
            receiver = ChatController.get_user_by_username(
                db, 
                receiver_username
            )
            if not receiver:
                await websocket.send_json({"error": "Receiver not found"})
                continue
            # Save message to DB
            ChatController.save_message(
                db, user.id, receiver.id, content
            )
            payload = {
                "from": username,
                "message": content
            }
            # Send to receiver if online
            if receiver_username in active_connections:
                await active_connections[receiver_username].send_json(payload)
            await websocket.send_json(payload) # Echo back to sender
    except WebSocketDisconnect:
        del active_connections[username]