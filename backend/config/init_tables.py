
import sys
import os

# Add the backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from config.database import init_db


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("✅ All tables created successfully!") 