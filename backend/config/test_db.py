import os
import sys
# Add the backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from config.database import engine
try:
    with engine.connect() as conn:
        print("✅ Connected to SQL Server!")
except Exception as e:
    print("❌ Connection failed:", e)