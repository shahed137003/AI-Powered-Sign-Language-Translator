from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

# paths - point to your model files
MODEL_PATH = ROOT / "models_features" / "best_model.pth"
LABEL_PATH = ROOT / "models_features" / "label_encoder.npy"

# preprocessing package
PREPROCESSING_PATH = ROOT / "Preprocessing_Landmarks"

# runtime config
DEVICE = torch.device("cpu")

TARGET_LEN = 120  # Change from 125 to 120 to match pipeline
COOLDOWN_FRAMES = 6
MIN_FRAMES = 8
CONF_THRESHOLD = 30

# Your model expects 928 features (after feature engineering)
MODEL_INPUT_DIM = 928