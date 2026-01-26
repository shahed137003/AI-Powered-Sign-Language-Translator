from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

import os
import sys

# Add the backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir=os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from config.settings import settings

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
# Create a base class for models
Base = declarative_base()

# Databse engine 
engine = create_engine(
  settings.DATABASE_URL,
  echo=True,
  pool_pre_ping=True,
  # fast_executemany=True
  pool_recycle=3600
)
# Session Local
SessionLocal = sessionmaker(
  autocommit = False,
  autoflush=False,
  bind =engine
)

# Dependency for FastAPI routes
def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# Initialize all tables
def init_db():
    from models.user import User
    from models.password_reset import PasswordReset
    # from models.translation import Translation
    Base.metadata.create_all(bind=engine)