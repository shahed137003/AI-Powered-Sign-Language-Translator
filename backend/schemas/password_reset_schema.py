from pydantic import BaseModel, EmailStr, Field

class ForgetPasswordSchema(BaseModel):
    email: EmailStr 
    
class ResetPasswordSchema(BaseModel):
    email: EmailStr
    code:str
    new_password: str 