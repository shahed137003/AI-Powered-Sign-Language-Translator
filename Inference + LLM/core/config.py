from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1] #get the path 2 levels up, thats root folder path

# paths
MODEL_PATH = ROOT / "models" / "model.pth"
LABEL_PATH = ROOT / "models" / "label_encoder.npy"

# preprocessing package
PREPROCESSING_PATH = ROOT / "Preprocessing_Landmarks"

# runtime config
DEVICE = torch.device("cpu")

TARGET_LEN = 120
COOLDOWN_FRAMES = 12 #how long to wait before ending gesture
MIN_FRAMES = 15 #used to ignore short gestures
CONF_THRESHOLD = 60 #used to ignore weak predictions, if confidence < 60 so not counted as valid word