from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from config.database import get_db
from models.user import User
from schemas.password_reset_schema import ForgetPasswordSchema, ResetPasswordSchema
from services.password_reset_service import create_reset_code, verify_reset_code
from services.email_service import send_email
from services.hashing import hash_password

router = APIRouter(prefix="/password", tags=["Password"])
@router.post("/forget")
def forget_password(data: ForgetPasswordSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    code = create_reset_code(user.id, db)
    send_email(user.email, "Password Reset Code", f"Your reset code is: {code}")
    return {"detail": "Reset code sent to your email"}
  
@router.post("/reset")
def reset_password(data: ResetPasswordSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    is_valid = verify_reset_code(user.id, data.code, db)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired reset code")
    user.password = hash_password(data.new_password)
    db.commit()
    return {"detail": "Password reset successful"}