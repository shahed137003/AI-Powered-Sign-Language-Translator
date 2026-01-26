from fastapi import HTTPException
from models.user import User, UserRole
from schemas.user_schema import UserCreate,AdminCreate
from services.hashing import hash_password

class UserController:
    # Normal user registration → always role=user
    @staticmethod
    def create_user(db, user_data:UserCreate):
        existing = db.query(User).filter(User.email == user_data.email).first()
        if existing:
            raise HTTPException(status_code=400,detail="Email already exists")
        hashed_pw = hash_password(user_data.password)
        new_user = User(
            username = user_data.username,
            email = user_data.email,
            password = hashed_pw,
            role = UserRole.user
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    
    # Admin creates admin
    @staticmethod
    def create_admin(db, admin_data:AdminCreate):
        existing = db.query(User).filter(User.email == admin_data.email).first()
        if existing:
            raise HTTPException(status_code= 400,detail="Email already exists")
        hashed_pw = hash_password(admin_data.password)
        new_admin=User(
            username = admin_data.username,
            email = admin_data.email,
            password = hashed_pw,
            role = UserRole.admin
        )
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
        return new_admin
    
    # Get user by ID (admin only)
    @staticmethod
    def get_user_by_id(db, user_id:int):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404,detail="User not found")
        return user
    
    # Get user by username (admin only)
    def get_user_by_username(db,username:str):
        user = db.query(User).filter(User.username==username).first()
        if not user:
            raise HTTPException(status_code=404,detail="User not found")
        return user
    
    # Get all users (admin only)
    def get_all_users(db):
        return db.query(User).all()
        