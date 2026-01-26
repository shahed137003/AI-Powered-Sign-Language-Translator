from datetime import datetime,timedelta
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi import Depends, HTTPException,status
from services.hashing import verify_password
from config.settings import settings
from sqlalchemy.orm import Session
from config.database import get_db
from models.user import User

# JWT CONFIG
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# SECURITY SCHEME FOR SWAGGER
auth_scheme = APIKeyHeader(name="Authorization")

# Create JWT token
def create_access_token(data:dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp":expire})
    return jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)
  
  
# Authenticate user on login
def authenticate_user(db, email: str, password: str, UserModel):
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if not user:
        raise HTTPException(status_code=400,detail="Invalid email")
    if not verify_password(password,user.password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    return user

# Extract current user from JWT
def get_current_user(
    raw_token:str =Depends(auth_scheme),
    db:Session = Depends(get_db)
):
    # Expect: "Bearer <token>"
    if not raw_token.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid token format. Use: Bearer <token>"
        )
    token = raw_token.split(" ")[1]  # Extract JWT only
    try:
        # Decode token 
        payload = jwt.decode(token ,SECRET_KEY,algorithms=[ALGORITHM])
        email:str = payload.get("sub")
        role:str = payload.get("role")
        
        if email is None or role is None:
            raise HTTPException(
                status_code=401,detail="Invalid authentication token"
            )
            
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Token is invalid or expired")
    
    # check user exists
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )
    return user