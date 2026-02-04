import mediapipe as mp
import cv2
import numpy as np
import itertools

# --- 1. SETUP MEDIA-PIPE ---
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # <--- FIX: Import face_mesh separately

# --- 2. DEFINE CONSTANTS ---
# We access the constants from 'mp_face_mesh', NOT 'mp_holistic'
FACEMESH_LIPS = set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))
FACEMESH_LEFT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW))
FACEMESH_RIGHT_EYEBROW = set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))

# Combine them into a single sorted list of unique indices
RELEVANT_FACE_INDICES = list(FACEMESH_LIPS | FACEMESH_LEFT_EYEBROW | FACEMESH_RIGHT_EYEBROW)
RELEVANT_FACE_INDICES.sort()

# Initialize the model
holistic = mp_holistic.Holistic(
    static_image_mode=False,       # False for video processing
    model_complexity=1,            # 1 is a good balance for accuracy/speed
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(image):
    """Accepts an image (numpy array) and returns a flat keypoint array."""
    # MediaPipe requires RGB
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    
    # 1. Pose (33 landmarks * 4 values [x,y,z,vis] = 132)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    
    # 2. Left Hand (21 landmarks * 3 values [x,y,z] = 63)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # 3. Right Hand (21 landmarks * 3 values [x,y,z] = 63)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    # 4. Face (Optimized: Only Lips & Eyebrows)
    if results.face_landmarks:
        # Extract only the relevant landmarks
        relevant_landmarks = [results.face_landmarks.landmark[i] for i in RELEVANT_FACE_INDICES]
        face = np.array([[res.x, res.y, res.z] for res in relevant_landmarks]).flatten()
    else:
        # Create zeros of the exact shape we need
        face = np.zeros(len(RELEVANT_FACE_INDICES) * 3)
        
    return np.concatenate([pose, face, lh, rh])