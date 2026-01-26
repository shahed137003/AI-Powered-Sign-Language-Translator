from typing import Optional
from pydantic import BaseModel, ConfigDict,EmailStr
from enum import Enum

class UserRole(str,Enum):
    admin = "admin"
    user = "user"
    
class UserCreate(BaseModel):
    username:str
    email:EmailStr
    password:str
    # role : UserRole=UserRole.user # role removed (forced as user)
   
   
class AdminCreate(BaseModel):
    username:str
    email:EmailStr
    password:str
     
class UserOut(BaseModel):
    id:int
    username : str
    email:EmailStr
    role:UserRole
    
    model_config = ConfigDict(from_attributes=True)  # Instead of orm_mode

class LogicSchema(BaseModel):
    email:EmailStr
    password:str
    
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None