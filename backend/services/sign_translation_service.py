# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# import joblib
# from collections import deque

# FRAME_LEN = 96

# class SignTranslationService:
#     def __init__(self):
#         self.model = tf.keras.models.load_model("../../ai/models/final_model.keras")
#         self.encoder = joblib.load("../../ai/models/label_encoder.joblib")
#         self.sequence = deque(maxlen=FRAME_LEN)

#         self.mp_holistic = mp.solutions.holistic
#         self.holistic = self.mp_holistic.Holistic(
#             min_detection_confidence=0.15,
#             min_tracking_confidence=0.15
#         )
#     def extract_keypoints(self, results):
#         keypoints = []

#         # Pose (33 * 3)
#         if results.pose_landmarks:
#             for lm in results.pose_landmarks.landmark:
#                 keypoints.extend([lm.x, lm.y, lm.z])
#         else:
#             keypoints.extend([0.0] * 99)

#         # Left Hand
#         if results.left_hand_landmarks:
#             for lm in results.left_hand_landmarks.landmark:
#                 keypoints.extend([lm.x, lm.y, lm.z])
#         else:
#             keypoints.extend([0.0] * 63)

#         # Right Hand
#         if results.right_hand_landmarks:
#             for lm in results.right_hand_landmarks.landmark:
#                 keypoints.extend([lm.x, lm.y, lm.z])
#         else:
#             keypoints.extend([0.0] * 63)

#         # Face (71 landmarks)
#         if results.face_landmarks:
#             for lm in results.face_landmarks.landmark[:71]:
#                 keypoints.extend([lm.x, lm.y, lm.z])
#         else:
#             keypoints.extend([0.0] * 213)

#         return np.array(keypoints, dtype=np.float32)

#     def process_frame(self, frame):
#         results = self.holistic.process(frame)
#         keypoints = self.extract_keypoints(results)
#         self.sequence.append(keypoints)

#         if len(self.sequence) < FRAME_LEN:
#             return None

#         x = np.expand_dims(self.sequence, axis=0)
#         probs = self.model.predict(x, verbose=0)[0]
#         idx = int(np.argmax(probs))

#         return {
#             "label": self.encoder.inverse_transform([idx])[0],
#             "confidence": float(probs[idx])
#         }