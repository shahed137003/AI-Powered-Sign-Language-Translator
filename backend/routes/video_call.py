from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from config.settings import settings
from config.database import SessionLocal
from models.user import User

router = APIRouter()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"


# ==============================
# Video Call Manager
# ==============================

class VideoCallManager:

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_calls: Dict[str, dict] = {}

    # --------------------------
    # Connection Management
    # --------------------------

    async def connect(self, websocket: WebSocket, username: str):
        await websocket.accept()
        self.active_connections[username] = websocket

    async def disconnect(self, username: str):

        if username in self.active_connections:
            del self.active_connections[username]

        # terminate any calls involving this user
        calls_to_end = []

        for call_id, call in self.active_calls.items():
            if call["caller"] == username or call["receiver"] == username:
                other = (
                    call["receiver"]
                    if call["caller"] == username
                    else call["caller"]
                )
                calls_to_end.append((call_id, other))

        for call_id, other_user in calls_to_end:

            await self.send_to_user(other_user, {
                "type": "call-ended",
                "callId": call_id,
                "reason": "peer-disconnected"
            })

            del self.active_calls[call_id]

    # --------------------------
    # Safe Send
    # --------------------------

    async def send_to_user(self, username: str, message: dict):

        if username not in self.active_connections:
            return

        try:
            await self.active_connections[username].send_json(message)
        except:
            # remove broken connection
            del self.active_connections[username]

    # --------------------------
    # Call Management
    # --------------------------

    def is_user_in_call(self, username: str):

        for call in self.active_calls.values():
            if call["caller"] == username or call["receiver"] == username:
                return True

        return False

    def create_call(self, call_id: str, caller: str, receiver: str, call_type: str):

        self.active_calls[call_id] = {
            "call_id": call_id,
            "caller": caller,
            "receiver": receiver,
            "call_type": call_type,
            "status": "ringing",
            "started_at": None
        }

        return self.active_calls[call_id]

    def get_call(self, call_id: str):

        return self.active_calls.get(call_id)

    def update_call_status(self, call_id: str, status: str):

        if call_id in self.active_calls:
            self.active_calls[call_id]["status"] = status

            if status == "connected":
                self.active_calls[call_id]["started_at"] = datetime.utcnow().isoformat()

    def end_call(self, call_id: str):

        if call_id in self.active_calls:
            del self.active_calls[call_id]
            return True

        return False


call_manager = VideoCallManager()


# ==============================
# WebSocket Endpoint
# ==============================

@router.websocket("/ws/video-call")
async def video_call_websocket(websocket: WebSocket):

    token = websocket.query_params.get("token")

    if not token:
        await websocket.close(code=1008)
        return

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")

    except JWTError:
        await websocket.close(code=1008)
        return

    db: Session = SessionLocal()

    user = db.query(User).filter(User.email == email).first()

    db.close()

    if not user:
        await websocket.close()
        return

    username = user.username

    await call_manager.connect(websocket, username)

    try:

        while True:

            data = await websocket.receive_json()

            # limit message size
            if len(str(data)) > 10000:
                await websocket.close(code=1009)
                return

            message_type = data.get("type")

            # --------------------------
            # Ping / Pong
            # --------------------------

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # --------------------------
            # CALL OFFER
            # --------------------------

            if message_type == "call-offer":

                target = data.get("target")
                offer = data.get("offer")
                call_id = data.get("callId", str(uuid.uuid4()))
                call_type = data.get("callType", "video")

                if target == username:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Cannot call yourself"
                    })
                    continue

                if target not in call_manager.active_connections:
                    await websocket.send_json({
                        "type": "user-offline",
                        "target": target
                    })
                    continue

                if call_manager.is_user_in_call(username):
                    await websocket.send_json({
                        "type": "error",
                        "message": "You are already in a call"
                    })
                    continue

                call_manager.create_call(call_id, username, target, call_type)

                await call_manager.send_to_user(target, {
                    "type": "incoming-call",
                    "callId": call_id,
                    "caller": username,
                    "callType": call_type,
                    "offer": offer
                })

            # --------------------------
            # CALL ANSWER
            # --------------------------

            elif message_type == "call-answer":

                target = data.get("target")
                answer = data.get("answer")
                call_id = data.get("callId")

                call = call_manager.get_call(call_id)

                if not call:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid call"
                    })
                    continue

                call_manager.update_call_status(call_id, "connected")

                await call_manager.send_to_user(target, {
                    "type": "call-answered",
                    "answer": answer,
                    "callId": call_id,
                    "from": username
                })

            # --------------------------
            # ICE CANDIDATE
            # --------------------------

            elif message_type == "ice-candidate":

                target = data.get("target")
                candidate = data.get("candidate")
                call_id = data.get("callId")

                call = call_manager.get_call(call_id)

                if not call:
                    continue

                await call_manager.send_to_user(target, {
                    "type": "ice-candidate",
                    "candidate": candidate,
                    "callId": call_id,
                    "from": username
                })

            # --------------------------
            # CALL REJECT
            # --------------------------

            elif message_type == "call-reject":

                target = data.get("target")
                call_id = data.get("callId")

                call_manager.end_call(call_id)

                await call_manager.send_to_user(target, {
                    "type": "call-rejected",
                    "callId": call_id,
                    "from": username
                })

            # --------------------------
            # CALL END
            # --------------------------

            elif message_type == "call-end":

                target = data.get("target")
                call_id = data.get("callId")

                call_manager.end_call(call_id)

                if target:
                    await call_manager.send_to_user(target, {
                        "type": "call-ended",
                        "callId": call_id,
                        "from": username
                    })

    except WebSocketDisconnect:

        await call_manager.disconnect(username)