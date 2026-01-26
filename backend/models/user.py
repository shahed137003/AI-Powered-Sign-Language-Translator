from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
import enum

from config.database import Base

class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True,index=True)
    username = Column(String(50),nullable=True)
    email = Column(String(100),nullable=False,unique=True,index=True)
    password = Column(String(255),nullable=False)
    role = Column(Enum(UserRole),default=UserRole.user)
    
    created_at = Column(DateTime(timezone=True),server_default=func.now())
    updated_at = Column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now())