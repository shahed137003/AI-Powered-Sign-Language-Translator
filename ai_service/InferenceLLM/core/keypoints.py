import mediapipe as mp
import numpy as np
import itertools

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face_mesh = mp.solutions.face_mesh
#we only use the lips and eyebrows from face
FACE_IDXS = sorted(list(
    set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)) |
    set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)) |
    set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW))
))


def extract_keypoints(results): #extracts the pose, face, left hand and right hand keypoints
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

    return np.concatenate([pose, face, lh, rh]) #returns one feature vector per frame