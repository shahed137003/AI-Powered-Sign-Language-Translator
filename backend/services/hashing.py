from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["argon2"],deprecated ="auto")

def hash_password(password:str):
    safe_password= password[:72]
    return pwd_context.hash(safe_password)

def verify_password(plain_password,hashed_password):
    return pwd_context.verify(plain_password[:72],hashed_password)
