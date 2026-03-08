import cv2
import numpy as np
import base64
import torch
import sys
import itertools
from collections import deque
from typing import Dict
from fastapi import WebSocket

# --- IMPORT YOUR PYTORCH TCN & UTILS ---
from asl_utils import config, preprocess_sequence_global, hybrid_frame_strategy
from train_model import TCN, FEATURE_DIM

# --- SAFE MEDIAPIPE IMPORT ---
import mediapipe as mp
try:
    from mediapipe.solutions import holistic as mp_holistic
    from mediapipe.solutions import face_mesh as mp_face_mesh
    from mediapipe.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    from mediapipe.python.solutions import holistic as mp_holistic
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- SETUP FACE INDICES ---
FACEMESH_LIPS = set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))
FACEMESH_LEFT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW))
FACEMESH_RIGHT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))
RELEVANT_FACE_INDICES = list(FACEMESH_LIPS | FACEMESH_LEFT_EYEBROW | FACEMESH_RIGHT_EYEBROW)
RELEVANT_FACE_INDICES.sort()

# --- LOAD PYTORCH MODEL ---
DEVICE = "cpu"
MODEL_PATH = "tcn_best_cpu.pth"
LABEL_PATH = "label_encoder.npy"

try:
    LABELS = np.load(LABEL_PATH, allow_pickle=True)
    num_classes = len(LABELS)
    model = TCN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ PyTorch TCN Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure tcn_best_cpu.pth and label_encoder.npy are in the ai_service folder.")
    sys.exit(1)


class SignToTextService:
    TARGET_FRAMES = 157
    MIN_FRAMES = 20

    def __init__(self, model, labels, device):
        self.model = model
        self.labels = labels
        self.device = device
        self.sequence = []

    def reset(self):
        self.sequence = []

    def process_keypoints(self, keypoints: list):
        """Accepts a flat list/array of keypoints from frontend."""
        kp_array = np.array(keypoints)
        self.sequence.append(kp_array)

        if len(self.sequence) >= self.TARGET_FRAMES:
            return self.predict_sequence()

        return {
            "status": "collecting",
            "frames_collected": len(self.sequence),
            "progress": int((len(self.sequence) / self.TARGET_FRAMES) * 100)
        }

    def predict_sequence(self):
        if len(self.sequence) < self.MIN_FRAMES:
            return {"text": "Too short", "confidence": 0.0}

        raw_sequence = np.array(self.sequence)
        cleaned = preprocess_sequence_global(raw_sequence)
        sequences, masks, metadata = hybrid_frame_strategy(cleaned, len(self.sequence))

        all_probs = []
        for seq, mask in zip(sequences, masks):
            x = torch.from_numpy(seq).float().unsqueeze(0).transpose(1, 2).to(self.device)
            m = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x, m)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[0].cpu().numpy())

        if len(all_probs) == 0:
            return {"text": "Invalid sequence", "confidence": 0.0}

        mean_probs = np.mean(np.stack(all_probs), axis=0)
        top_idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[top_idx])
        predicted_word = self.labels[top_idx]

        self.sequence = []
        return {"text": predicted_word, "confidence": confidence}