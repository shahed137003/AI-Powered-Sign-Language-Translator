from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

import os
import sys

# Add the backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir=os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from config.settings import settings

# Create a base class for models
Base = declarative_base()

# Database engine
# NullPool is used with pyodbc/MSSQL to prevent connection reuse issues.
# pyodbc connections can silently carry over failed transaction state across
# pool reuse, causing all subsequent requests to fail until server restart.
# NullPool gives every request a brand-new connection, completely avoiding this.
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,         # set True only for debugging SQL queries
    poolclass=NullPool, # no connection reuse — fresh connection every request
    connect_args={
        "timeout": 30,      # connection timeout in seconds
        "autocommit": False # explicit transaction control
    }
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        
# Initialize all tables
def init_db():
    from models.user import User
    from models.password_reset import PasswordReset
    # from models.translation import Translation
    Base.metadata.create_all(bind=engine)