import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import itertools
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "InferenceLLM"))
sys.path.insert(0, str(current_dir / "InferenceLLM" / "Preprocessing_Landmarks"))

from preprocessing.pipeline_v3 import preprocess_sequence_global
from preprocessing.constants import FEATURE_DIM
from preprocessing.features.builder import build_features
from core.llm import LLMTranslator

print("=" * 60)
print("ASL DEBUGGING TOOL")
print("=" * 60)

# ============================================================
# MODEL ARCHITECTURE - SAME AS YOUR core/model.py
# ============================================================
class TemporalBlock(nn.Module):
    def __init__(self, inp, out, dil, drop):
        super().__init__()
        p = dil
        self.net = nn.Sequential(
            nn.Conv1d(inp, out, 3, padding=p, dilation=dil),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(out, out, 3, padding=p, dilation=dil),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.res = nn.Identity() if inp == out else nn.Conv1d(inp, out, 1)

    def forward(self, x):
        y = self.net(x)
        if y.size(-1) != x.size(-1):
            y = y[..., :x.size(-1)]
        return y + self.res(x)


class TCN(nn.Module):
    def __init__(self, input_dim, num_classes, ch=128, layers=4, drop=0.3):
        super().__init__()
        blocks = []
        for i in range(layers):
            ind = input_dim if i == 0 else ch
            blocks.append(TemporalBlock(ind, ch, 2**i, drop))
        self.tcn = nn.Sequential(*blocks)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(ch, num_classes)

    def masked_pool(self, x, m):
        m = m.unsqueeze(1)
        x = x * m
        s = x.sum(-1)
        d = m.sum(-1).clamp(min=1)
        return s / d

    def forward(self, x, m):
        x = self.tcn(x)
        x = self.masked_pool(x, m)
        x = self.drop(x)
        return self.fc(x)


# Load model
DEVICE = torch.device("cpu")
MODEL_PATH = current_dir / "InferenceLLM" / "models_features" / "best_model.pth"
LABEL_PATH = current_dir / "InferenceLLM" / "models_features" / "label_encoder.npy"

print(f"\n📁 Model: {MODEL_PATH}")
print(f"📁 Labels: {LABEL_PATH}")

# Load labels - handle dictionary format
try:
    LABELS_DATA = np.load(LABEL_PATH, allow_pickle=True)
    
    if isinstance(LABELS_DATA, np.ndarray) and LABELS_DATA.ndim == 0:
        LABELS_DATA = LABELS_DATA.item()
    
    if isinstance(LABELS_DATA, dict):
        class_names = list(LABELS_DATA.keys())
        sorted_classes = sorted(class_names, key=lambda x: LABELS_DATA[x])
        LABELS = np.array(sorted_classes)
        num_classes = len(LABELS)
        print(f"✅ Loaded {num_classes} classes from dictionary")
        print(f"   First 10: {LABELS[:10]}")
    else:
        LABELS = np.array(LABELS_DATA)
        num_classes = len(LABELS)
        print(f"✅ Loaded {num_classes} classes from array")
        print(f"   First 10: {LABELS[:10]}")
        
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    sys.exit(1)

# Model expects 928 features
MODEL_INPUT_DIM = 928
model = TCN(input_dim=MODEL_INPUT_DIM, num_classes=num_classes, ch=128, layers=4, drop=0.3).to(DEVICE)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model loaded\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Initialize LLM Translator
translator = None
try:
    translator = LLMTranslator()
    print("✅ LLM Translator ready\n")
except Exception as e:
    print(f"⚠️ Could not initialize LLM Translator (GROQ_API_KEY might be missing): {e}\n")

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
FACE_IDXS = sorted(list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)) | 
                        set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)) | 
                        set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))))

def get_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in FACE_IDXS]).flatten() if results.face_landmarks else np.zeros(180)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def predict_with_debug(buffer_frames):
    """Run prediction with detailed debug output"""
    raw_seq = np.array(buffer_frames)
    print(f"\n{'='*50}")
    print(f"Raw sequence shape: {raw_seq.shape}")
    print(f"Raw - min: {raw_seq.min():.3f}, max: {raw_seq.max():.3f}, mean: {raw_seq.mean():.3f}")
    
    # Step 1: Preprocess
    proc_seq = preprocess_sequence_global(raw_seq)
    print(f"After preprocess shape: {proc_seq.shape}")
    print(f"Preproc - min: {proc_seq.min():.3f}, max: {proc_seq.max():.3f}, mean: {proc_seq.mean():.3f}")
    
    # Step 2: Feature engineering
    features = build_features(proc_seq)
    print(f"After features shape: {features.shape}")
    print(f"Features - min: {features.min():.3f}, max: {features.max():.3f}, mean: {features.mean():.3f}")
    
    # Step 3: Pad/trim
    TARGET_LEN = 120
    T = features.shape[0]
    if T > TARGET_LEN:
        final_input = features[:TARGET_LEN]
    else:
        pad = np.zeros((TARGET_LEN - T, features.shape[1]), dtype=np.float32)
        final_input = np.concatenate([features, pad], axis=0)
    print(f"Final input shape: {final_input.shape}")
    
    # Step 4: Create mask (which frames are real vs padded)
    mask = np.ones(TARGET_LEN, dtype=np.float32)
    if T < TARGET_LEN:
        mask[T:] = 0
    
    # Step 5: Predict - transpose to (Batch, Channels, Time)
    x = torch.from_numpy(final_input).float().transpose(0, 1).unsqueeze(0).to(DEVICE)
    m = torch.from_numpy(mask).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x, m)
        probs = torch.softmax(logits, dim=1)
        top5_idx = torch.topk(probs, 5, dim=1).indices[0].cpu().numpy()
        top5_conf = torch.topk(probs, 5, dim=1).values[0].cpu().numpy()
    
    print(f"\n📊 Top 5 predictions:")
    for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
        print(f"   {i+1}. {LABELS[idx]}: {conf*100:.1f}%")
    
    top_pred = LABELS[top5_idx[0]]
    top_conf = top5_conf[0] * 100
    
    return top_pred, top_conf

# Webcam loop
cap = cv2.VideoCapture(0)
buffer = []
sentence_buffer = []
recording = False
silence_counter = 0
COOLDOWN_FRAMES = 12
MIN_FRAMES = 15

print("\n🎥 Webcam started. Show hands to record, hide to predict.")
print("Press 'q' to quit, 'r' to reset buffer\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    
    hand_seen = results.left_hand_landmarks or results.right_hand_landmarks
    kp = get_keypoints(results)
    
    if hand_seen:
        if not recording:
            recording = True
            buffer = []
            print("\n🔴 RECORDING started...")
        buffer.append(kp)
        silence_counter = 0
    elif recording:
        silence_counter += 1
        buffer.append(kp)
        
        if silence_counter > COOLDOWN_FRAMES:
            recording = False
            print(f"\n🟢 Recording stopped. Frames: {len(buffer)}")
            
            if len(buffer) > MIN_FRAMES:
                pred, conf = predict_with_debug(buffer)
                
                # Check confidence threshold (30.0%)
                if conf >= 30.0:
                    if not sentence_buffer or sentence_buffer[-1] != pred:
                        sentence_buffer.append(pred)
                        print(f"📝 Added to sentence buffer: {sentence_buffer}")
                else:
                    print(f"⚠️ Low confidence ({conf:.1f}%), not adding to sentence buffer.")
                
                # Translate accumulated sentence buffer
                if sentence_buffer:
                    gloss_sentence = " ".join(sentence_buffer)
                    if translator:
                        print(f"\n💬 Translating sentence buffer '{gloss_sentence}' with LLM...")
                        success, translation = translator.translate(gloss_sentence)
                        if success:
                            print(f"✨ LLM Sentence Translation: {translation}")
                        else:
                            print(f"❌ LLM Translation failed: {translation}")
                else:
                    print("⚠️ Sentence buffer is empty.")
            else:
                print(f"⚠️ Too few frames: {len(buffer)} (need {MIN_FRAMES})")
            buffer = []
    
    # UI
    if recording:
        cv2.circle(frame, (40, 40), 15, (0, 0, 255), -1)
        cv2.putText(frame, f"RECORDING ({len(buffer)})", (70, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("ASL Debug", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        sentence_buffer = []
        print("\n🔄 Sentence buffer cleared!")

cap.release()
cv2.destroyAllWindows()