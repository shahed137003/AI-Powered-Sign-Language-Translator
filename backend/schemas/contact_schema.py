from pydantic import BaseModel, EmailStr ,ConfigDict
class ContactCreate(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str
    
class ContactOut(BaseModel):
    id: int
    user_id: int
    name: str
    email: EmailStr
    subject: str
    message: str
    
    model_config = ConfigDict(from_attributes=True)