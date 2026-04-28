from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from schemas.contact_schema import ContactCreate, ContactOut
from controllers.contact_controller import ContactController
from config.database import get_db
from models.user import User,UserRole
from services.auth import get_current_user
from services.email_service import send_email
from config.settings import settings

router = APIRouter(prefix="/contacts", tags=["Contacts"])

@router.post("/send", response_model=ContactOut)
def send_message(
    data: ContactCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Save message in DB
    new_message = ContactController.create_message(
      db, 
      data,
      user_id=current_user.id
    )
    
    # Email Body for admin
    admin_body = (
        f"New contact message received:\n\n"
        f"From Name: {data.name}\n"
        f"From Email: {data.email}\n"
        f"User ID: {current_user.id}\n"
        f"Subject: {data.subject}\n\n"
        f"Message:\n{data.message}\n\n"
        f"Message ID: {new_message.id}"
    )
    # Send email to admin
    send_email(
        settings.ADMIN_EMAIL,
        f"New Contact Message {data.subject}",
        admin_body
    )
    return new_message
    

@router.get("/all", response_model=list[ContactOut])
def get_all_messages(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Not authorized to view all messages")  
    return ContactController.get_all_messages(db)

@router.get("/by-id/{message_id}", response_model=ContactOut)
def get_by_id(
    message_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role == UserRole.admin:
        # Admin can see any message
        return ContactController.get_message_admin(db, message_id)
    # Normal user can see only their own messages
    return ContactController.get_message_by_id(db, message_id, current_user.id)

@router.get("/by-name/{name}", response_model=list[ContactOut])
def get_by_name(
    name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Not authorized to view messages by name")  
    return ContactController.get_by_username(db, name)