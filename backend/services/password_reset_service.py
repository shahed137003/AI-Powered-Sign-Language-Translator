from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.password_reset import PasswordReset
from random import randint

def generate_code():
    return f"{randint(100000, 999999)}"
  
def create_reset_code(user_id: int, db: Session)-> str:
    code = generate_code()
    expires_at = datetime.utcnow() + timedelta(minutes=15)
    reset_entry = PasswordReset(user_id=user_id, code=code, expires_at=expires_at)
    db.add(reset_entry)
    db.commit()
    db.refresh(reset_entry)
    return code
def verify_reset_code(user_id: int, code: str, db: Session):
    reset_entry = db.query(PasswordReset).filter(
        PasswordReset.user_id == user_id,
        PasswordReset.code == code,
        PasswordReset.expires_at >= datetime.utcnow()
    ).first()
    if not reset_entry:
        return False
    db.delete(reset_entry)
    db.commit()
    return True
    