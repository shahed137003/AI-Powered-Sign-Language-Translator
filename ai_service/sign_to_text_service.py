import cv2
import numpy as np
import base64
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque
import time
MODEL_PATH = "../ai/models/final_model.keras"
ENCODER_PATH = "../ai/models/label_encoder.joblib"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully")
    label_encoder = joblib.load(ENCODER_PATH)
    print("✓ Label encoder loaded successfully")
    print(f"Classes (first 10): {label_encoder.classes_[:10]}")
except Exception as e:
    print(f"✗ Error: {e}")
    raise

FRAME_LEN = 96

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility]
                  for lm in results.pose_landmarks.landmark])
        if results.pose_landmarks
        else np.zeros((33, 4))
    )

    lh = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.left_hand_landmarks.landmark])
        if results.left_hand_landmarks
        else np.zeros((21, 3))
    )

    rh = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.right_hand_landmarks.landmark])
        if results.right_hand_landmarks
        else np.zeros((21, 3))
    )

    face = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.face_landmarks.landmark[:60]])
        if results.face_landmarks
        else np.zeros((60, 3))
    )

    return np.concatenate([
        pose.flatten(),   # 132
        lh.flatten(),     # 63
        rh.flatten(),     # 63
        face.flatten()    # 180
    ])

class SignToTextService:
    def __init__(self):
        self.sequence = deque(maxlen=FRAME_LEN)
        # Direct instantiation
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    
    def process_frame(self, base64_image: str):
        print("📩 Frame received, sequence length:", len(self.sequence))
        try:
            if "," in base64_image:
                base64_image = base64_image.split(",")[1]
            
            image_bytes = base64.b64decode(base64_image)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("❌ Error: Frame decoded is None (Empty image)")
                return {"status": "error", "text": "Empty Frame"}
            
            # 2. Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.flip(rgb, 1)
            results = self.holistic.process(rgb)
            
            # --- DEBUGGING: CHECK IF LANDMARKS EXIST ---
            has_any_landmark = (
                results.pose_landmarks or
                results.left_hand_landmarks or
                results.right_hand_landmarks or
                results.face_landmarks
            )

            if not has_any_landmark:
                print("⚠️ No landmarks at all")
                self.sequence.append(np.zeros(438, dtype=np.float32))

                frames_collected = len(self.sequence)
                progress_percentage = (frames_collected / FRAME_LEN) * 100

                return {
                    "status": "collecting",
                    "text": f"Waiting for movement... {frames_collected}/{FRAME_LEN}",
                    "confidence": 0.0,
                    "progress": progress_percentage,
                    "frames_collected": frames_collected
                }
            if len(self.sequence) == 0 and has_any_landmark:
                cv2.imwrite("debug_ws_frame.jpg", frame)
                print("📸 Saved debug_ws_frame.jpg")

            # --- VISUALIZATION: DRAW LANDMARKS & SAVE TEST IMAGE ---
            # Only save the first valid frame to verify it looks right
            if len(self.sequence) == 0:
                debug_image = frame.copy()
                mp_drawing.draw_landmarks(debug_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(debug_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(debug_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                # Save to disk to check manually
                cv2.imwrite(f"debug_frame_{int(time.time())}.jpg", debug_image)
                print(f"📸 Debug image saved as debug_frame_{int(time.time())}.jpg")
            
            keypoints = extract_keypoints(results)
            print(
                f"📐 KP shape: {keypoints.shape} | "
                f"Sum: {np.sum(keypoints):.2f} | "
                f"Pose: {bool(results.pose_landmarks)} | "
                f"LH: {bool(results.left_hand_landmarks)} | "
                f"RH: {bool(results.right_hand_landmarks)}"
            )
            print("📦 Sequence length BEFORE:", len(self.sequence))
            self.sequence.append(keypoints)
            print("📦 Sequence length AFTER:", len(self.sequence))
            
            frames_collected = len(self.sequence)
            progress_percentage = (frames_collected / FRAME_LEN) * 100
            
            if frames_collected == FRAME_LEN:
                x = np.expand_dims(list(self.sequence), axis=0)
                
                # x = np.expand_dims(np.array(self.sequence), axis=0)
                
                print("🧠 Model input shape:", x.shape)
                probs = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                
                return {
                    "text": label_encoder.inverse_transform([idx])[0],
                    "confidence": confidence,
                    "progress": 100,
                    "frames_collected": frames_collected
                }
            
            # Return progress information when collecting frames
            return {
                "text": f"Collecting frames: {frames_collected}/{FRAME_LEN}",
                "confidence": 0.0,
                "progress": progress_percentage,
                "frames_collected": frames_collected
            }
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return {
                "text": "error",
                "confidence": 0.0,
                "progress": 0,
                "frames_collected": 0
            }