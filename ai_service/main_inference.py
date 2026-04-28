import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import itertools
import time
from pathlib import Path

# Import the specific pipeline functions from your friend's preprocessing package
from preprocessing.pipeline_v3 import preprocess_sequence_global
from preprocessing.constants import FEATURE_DIM

# ============================================================
# 1. MODEL ARCHITECTURE (Matches Training Script)
# ============================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., :x.size(2)]
        return y + self.res(x)

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        channels = [128, 128, 128, 128] 
        layers = []
        for i, c in enumerate(channels):
            in_dim = input_dim if i == 0 else channels[i-1]
            layers.append(TemporalBlock(in_dim, c, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.tcn(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# ============================================================
# 2. CONFIGURATION & LOADING
# ============================================================
DEVICE = torch.device("cpu")
MODEL_PATH = "Not cleaned + preprocessing no mask/TCN_preprocessed_no_cleaning_no_mask.pth"
# "/TCN_model/Not cleaned + preprocessing no mask/TCN_preprocessed_no_cleaning_no_mask.pth"
LABEL_PATH = "Not cleaned + preprocessing no mask/label_encoder.npy"
# "/TCN_model/Not cleaned + preprocessing no mask/label_encoder.npy"
SAVE_DIR = Path("recorded_npy_files")
SAVE_DIR.mkdir(exist_ok=True)

# Fixed settings based on training output
TARGET_LEN = 120 
COOLDOWN_FRAMES = 12  # How many frames of "no hands" before triggering prediction
MIN_FRAMES = 15       # Don't predict if the gesture was too short

# Load Mapping and Model
LABELS = np.load(LABEL_PATH, allow_pickle=True)
num_classes = len(LABELS)
model = TCN(input_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)

# Load the model weights from the saved checkpoint dictionary
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Extract relevant face indices (Lips and Eyebrows)
mp_face_mesh = mp.solutions.face_mesh
FACE_IDXS = sorted(list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)) | 
                        set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)) | 
                        set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))))

def get_kp(results):
    """Parses MediaPipe results into the raw 438-feature vector."""
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in FACE_IDXS]).flatten() if results.face_landmarks else np.zeros(180)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

# ============================================================
# 3. WEBCAM LOOP WITH AUTO-TRIGGER
# ============================================================
cap = cv2.VideoCapture(0)
buffer = []
recording = False
silence_counter = 0
last_pred = "Waiting..."
last_conf = 0.0

print(f"--- AUTO-TRIGGER ACTIVE ---")
print(f"Recording starts when hands are seen.")
print(f"Prediction triggers after hands are removed.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # Process MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    
    hand_seen = results.left_hand_landmarks or results.right_hand_landmarks
    kp = get_kp(results)

    # Trigger Logic
    if hand_seen:
        if not recording:
            recording = True
            buffer = []
            print("Hand Detected - Recording...")
        buffer.append(kp)
        silence_counter = 0
    elif recording:
        # Hand was just seen, but is gone now
        silence_counter += 1
        buffer.append(kp) # Add empty frames for the grace period
        
        if silence_counter > COOLDOWN_FRAMES:
            recording = False
            print(f"Hands gone. Buffer size: {len(buffer)}")
            
            if len(buffer) > MIN_FRAMES:
                # 1. Save raw data
                filename = f"gesture_{int(time.time())}.npy"
                np.save(SAVE_DIR / filename, np.array(buffer))
                
                # 2. Preprocess using v3 pipeline
                raw_seq = np.array(buffer)
                proc_seq = preprocess_sequence_global(raw_seq)
                
                # 3. Fixed Length Padding/Trimming to 120
                T = proc_seq.shape[0]
                if T > TARGET_LEN:
                    final_input = proc_seq[:TARGET_LEN]
                else:
                    final_input = np.concatenate([proc_seq, np.zeros((TARGET_LEN-T, FEATURE_DIM))])
                
                # 4. Inference
                # Model expects (Batch, Channels, Time) -> (1, 438, 120)
                x = torch.from_numpy(final_input).float().transpose(0, 1).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    
                last_pred = LABELS[idx.item()]
                last_conf = conf.item() * 100
                print(f"Result: {last_pred} ({last_conf:.1f}%) | Saved to: {filename}")
            
            buffer = [] # Reset

    # UI Visuals
    if recording:
        cv2.circle(frame, (40, 40), 15, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING", (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display Prediction Results
    color = (0, 255, 0) if last_conf > 60 else (0, 255, 255)
    cv2.putText(frame, f"Gesture: {last_pred}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {last_conf:.1f}%", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Real-Time ASL Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()