import sys
from pathlib import Path

# Ensure the parent directory (Preprocessing_Landmarks) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.cli_v3 import main

if __name__ == "__main__":
    main()
