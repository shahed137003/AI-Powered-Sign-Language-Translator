from fastapi import APIRouter,Depends ,HTTPException
from sqlalchemy.orm import Session

from schemas.user_schema import UserCreate,UserOut,LogicSchema,AdminCreate
from controllers.user_controller import UserController
from services.auth import authenticate_user,create_access_token,get_current_user
from config.database import get_db
from models.user import User,UserRole

router = APIRouter(prefix="/users",tags=["Users"])

# Public registration → always creates user
@router.post("/register",response_model=UserOut)
def register(user_data:UserCreate,db:Session = Depends(get_db)):
    return UserController.create_user(db,user_data)

# Only admin can create admin
@router.post("/create_admin",response_model=UserOut)
def create_admin(
    admin_data :AdminCreate,
    db:Session = Depends(get_db),
    current_user:User = Depends(get_current_user) 
):
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403,detail="Only admins can create admins")
    return UserController.create_admin(db,admin_data)

# Login
@router.post("/login")
def login(credentials:LogicSchema,db:Session=Depends(get_db)):
    user= authenticate_user(db,credentials.email,credentials.password,User)
    token = create_access_token({
        "sub":user.email,
        "id":user.id,
        "role":user.role
    })
    return {
        "access_token":token,
        "token_type":"bearer",
        "user":{
            "id":user.id,
            "username":user.username,
            "email":user.email,
            "role":user.role
        }
    }


# get current user info (critcal for frontend)
@router.get("/me",response_model=UserOut)
def get_current_user_info(current_user:User = Depends(get_current_user)):
    return current_user

# Admin only - get user by ID
@router.get("/get-by-id/{user_id}",response_model=UserOut)
def get_user_by_id(
    user_id:int,
    db:Session = Depends(get_db),
    current_user :User = Depends(get_current_user)
):
    # if current_user.role != UserRole.admin:
    #     raise HTTPException(status_code=403,detail="Admins only")
    return UserController.get_user_by_id(db,user_id)

# get user by username
@router.get("/get-by-username/{username}",response_model=UserOut)
def get_user_by_username(
    username:str,
    db:Session = Depends(get_db),
    current_user:User = Depends(get_current_user)
):
    return UserController.get_user_by_username(db,username)

# search users(for finding people to chat with)
@router.get("/search/{query}",response_model=list[UserOut])
def search_users(
    query:str,
    db:Session = Depends(get_db),
    current_user:User = Depends(get_current_user)
):
    users = db.query(User).filter(
        (User.username.ilike(f"%{query}%")) | 
        (User.email.ilike(f"%{query}%"))
    ).limit(10).all()
    return users

# Admin only - get all users 
@router.get("/all",response_model=list[UserOut])
def get_all_users(
    db:Session = Depends(get_db),
    current_user :User = Depends(get_current_user)
):
    # if current_user.role != UserRole.admin:
    #     raise HTTPException(status_code=403,detail="Admins only")
    return UserController.get_all_users(db)

# user can edit his name 
@router.put("/me",response_model=UserOut)
def update_profile(
    new_data:dict , 
    db:Session = Depends(get_db),
    current_user :User = Depends(get_current_user)
):
    if "username" in new_data:
        current_user.username = new_data["username"]
    db.commit()
    db.refresh(current_user)
    return current_user

@router.get("/check-username/{username}")
def check_username_exists(username: str, db: Session = Depends(get_db)):
    """Check if a username exists in the system"""
    user = db.query(User).filter(User.username == username).first()
    return {"exists": user is not None, "username": username}

@router.get("/chat-users",response_model=list[UserOut])
def get_chat_users(
    db:Session = Depends(get_db),
    current_user :User = Depends(get_current_user)
):
    users = db.query(User).filter(User.id != current_user.id).all()
    return users