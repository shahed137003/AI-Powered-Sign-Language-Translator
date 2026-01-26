from fastapi import HTTPException
from models.contact import Contact

class ContactController:
    @staticmethod
    def create_message(db,data, user_id:int):
        new_message = Contact(
            user_id=user_id,
            name=data.name,
            email=data.email,
            subject=data.subject,
            message=data.message
        )
        db.add(new_message)
        db.commit()
        db.refresh(new_message)
        return new_message
      
    @staticmethod
    def get_all_messages(db):
        return db.query(Contact).all()
      
    @staticmethod
    def get_message_by_id(db,message_id,user_id):
        message = db.query(Contact).filter(Contact.id == message_id, Contact.user_id == user_id).first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return message
      
    @staticmethod 
    def get_message_admin(db, message_id):
        message = db.query(Contact).filter(Contact.id == message_id).first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return message
    
    
    @staticmethod
    def get_by_username(db, name):
        message = db.query(Contact).filter(Contact.name == name).all()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return message
