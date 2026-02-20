from pathlib import Path

DATA_DIR = Path(r"E:\ASL_Citizen\NEW\Top_Classes_Landmarks_Preprocessed")

DEVICE = "cpu"

TARGET_FRAMES = 157
FEATURE_DIM = 438

BATCH_SIZE = 8
EPOCHS = 90
LR = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 12
GRAD_CLIP = 1.0
LABEL_SMOOTH = 0.1

CHECKPOINT_DIR = Path("experiments/checkpoints/146_gloss")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = CHECKPOINT_DIR / "tcn_best.pth"
LABEL_ENCODER_PATH = Path("data/processed/encoders/label_encoder.npy")