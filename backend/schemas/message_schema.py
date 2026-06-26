from pydantic import BaseModel, ConfigDict
from datetime import datetime

class MessageOut(BaseModel):
    id: int
    sender_id: int
    sender_username: str
    receiver_id: int
    receiver_username: str
    content: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)