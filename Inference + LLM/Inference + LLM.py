import sys
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import itertools
import time


# preprocessing
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "Preprocessing_Landmarks"))

from preprocessing.pipeline_v3 import preprocess_sequence_global
from preprocessing.constants import FEATURE_DIM

# ============================================================
# MODEL ARCHITECTURE
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
            in_dim = input_dim if i == 0 else channels[i - 1]
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
# CONFIG
# ============================================================
DEVICE = torch.device("cpu")

MODEL_PATH = "Not cleaned + preprocessing no mask/TCN_preprocessed_no_cleaning_no_mask.pth"
LABEL_PATH = "Not cleaned + preprocessing no mask/label_encoder.npy"

SAVE_DIR = Path("recorded_npy_files")
SAVE_DIR.mkdir(exist_ok=True)

TARGET_LEN = 120
COOLDOWN_FRAMES = 12
MIN_FRAMES = 15


LABELS = np.load(LABEL_PATH, allow_pickle=True)
num_classes = len(LABELS)

model = TCN(input_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# ============================================================
# MEDIA PIPE
# ============================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
FACE_IDXS = sorted(list(
    set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)) |
    set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)) |
    set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))
))


def get_kp(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    face = np.array([[results.face_landmarks.landmark[i].x,
                      results.face_landmarks.landmark[i].y,
                      results.face_landmarks.landmark[i].z]
                     for i in FACE_IDXS]).flatten() if results.face_landmarks else np.zeros(180)

    lh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    rh = np.array([[lm.x, lm.y, lm.z]
                   for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, face, lh, rh])


# ============================================================
# LLM
# ============================================================
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import nltk

nltk.download('wordnet')
nltk.download('punkt')

print("Loading LLM...")

model_path = hf_hub_download(
    repo_id="bartowski/gemma-2-9b-it-GGUF",
    filename="gemma-2-9b-it-Q4_K_M.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=6,
    n_batch=512,
    verbose=False
)


FULL_CONTEXT = (
    "You are an expert ASL-to-English translator. Translate ASL gloss into fluent, natural English. "
    "Follow these structural rules based on the sentence type:\n\n"
    "# 1. Declarative / Statements\n"
    "ASL: ME + [verb] + [object/time/location/emotion] → English: Use 'I' + verb + rest of sentence. "
    "Apply proper tense (FINISH/YESTERDAY → past, TOMORROW/NEXT WEEK → future).\n\n"
    "# 2. Questions (Yes/No)\n"
    "ASL: YOU + [verb] + [object/time/etc]? → English: Form a natural yes/no question. Invert subject-verb as needed.\n\n"
    "# 3. WH-Questions\n"
    "ASL: [WH-word] + [subject] + [verb] → English: Use correct wh-word placement (WHAT, WHERE, WHEN, WHO, WHY, HOW).\n\n"
    "# 4. Modal verbs / Permission / Ability / Obligation\n"
    "ASL: ME + [modal verb] + [verb] + [object/time] → English: Render modal naturally.\n\n"
    "# 5. Emotions / States\n"
    "ASL: ME + [emotion/state] + [time] → English: I am + emotion/state [+ time if provided].\n\n"
    "# 6. Past / Future events\n"
    "ASL: ME + [verb] + [object] + [time] → English: Translate verb tense according to time word.\n\n"
    "# 7. Rules for output\n"
    "1. Return ONLY the English sentence, no commentary.\n"
    "2. Use proper grammar, tense, and subject-verb agreement.\n"
    "3. Contractions are allowed for natural English.\n"
    "4. Preserve meaning; do not invent extra information.\n\n"
    "# Few-shot examples:\n"
    "ASL: ME HOME GO\nEnglish: I am going home.\n"
    "ASL: BOOK FINISH READ ME\nEnglish: I have read the book.\n"
    "ASL: YOU GO STORE TOMORROW?\nEnglish: Are you going to the store tomorrow?\n"
    "ASL: NAME YOU WHAT\nEnglish: What is your name?\n"
    "ASL: ME CAN SWIM\nEnglish: I can_swim.\n"
    "ASL: ME HAPPY TODAY\nEnglish: I am happy today.\n"
    "ASL: ME EAT PIZZA YESTERDAY\nEnglish: I ate pizza yesterday.\n"
    "ASL: ME GO PARK TOMORROW\nEnglish: I will go to the park tomorrow.\n"
)


def translate_gloss(gloss):
    prompt = f"<|im_start|>user\n{FULL_CONTEXT}ASL: {gloss}<|im_end|>\n<|im_start|>assistant\nEnglish:"
    
    output = llm(
        prompt,
        max_tokens=40,
        stop=["<|im_end|>", "\n", "ASL:"],
        temperature=0.0
    )
    return output["choices"][0]["text"].strip().replace('"', '')


# ============================================================
# STATE
# ============================================================
sentence_buffer = []
english_sentence = ""


# ============================================================
# WEBCAM LOOP
# ============================================================
cap = cv2.VideoCapture(0)

buffer = []
recording = False
silence_counter = 0

last_pred = "Waiting..."
last_conf = 0.0

print("Running...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    hand_seen = results.left_hand_landmarks or results.right_hand_landmarks
    kp = get_kp(results)

    # =========================
    # RECORDING LOGIC
    # =========================
    if hand_seen:
        if not recording:
            recording = True
            buffer = []

        buffer.append(kp)
        silence_counter = 0

    elif recording:
        silence_counter += 1
        buffer.append(kp)

        if silence_counter > COOLDOWN_FRAMES:
            recording = False

            if len(buffer) > MIN_FRAMES:

                raw_seq = np.array(buffer)
                proc_seq = preprocess_sequence_global(raw_seq)

                T = proc_seq.shape[0]
                if T > TARGET_LEN:
                    final_input = proc_seq[:TARGET_LEN]
                else:
                    final_input = np.concatenate(
                        [proc_seq, np.zeros((TARGET_LEN - T, FEATURE_DIM))]
                    )

                x = torch.from_numpy(final_input).float().transpose(0, 1).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)

                last_pred = LABELS[idx.item()]
                last_conf = conf.item() * 100

                # =========================
                # BUILD SENTENCE
                # =========================
                if last_conf > 60:
                    if len(sentence_buffer) == 0 or sentence_buffer[-1] != last_pred:
                        sentence_buffer.append(last_pred)

                buffer = []

                # =========================
                # TRIGGER LLM (FIXED)
                # =========================
                if len(sentence_buffer) >=3:
                    gloss = " ".join(sentence_buffer)
                    print("Gloss:", gloss)

                    english_sentence = translate_gloss(gloss)
                    print("English:", english_sentence)

                    sentence_buffer = []


    # =========================
    # UI
    # =========================
    cv2.putText(frame, f"Gesture: {last_pred}", (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {last_conf:.1f}%", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Words: {' '.join(sentence_buffer)}", (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"English: {english_sentence}", (20, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ASL System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
