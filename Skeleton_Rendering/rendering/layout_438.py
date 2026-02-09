import numpy as np

# ---------- layout (pose, face, LH, RH) ----------
POSE_LANDMARKS, POSE_VALS = 33, 4
HAND_LANDMARKS, HAND_VALS = 21, 3
FACE_LANDMARKS, FACE_VALS = 60, 3

POSE_SIZE = POSE_LANDMARKS * POSE_VALS          # 132
HAND_SIZE = HAND_LANDMARKS * HAND_VALS          # 63
FACE_SIZE = FACE_LANDMARKS * FACE_VALS          # 180
FEATURE_DIM = POSE_SIZE + FACE_SIZE + 2 * HAND_SIZE  # 438


def unpack_frame(vec: np.ndarray):
    pose_flat = vec[:POSE_SIZE]
    face_flat = vec[POSE_SIZE:POSE_SIZE + FACE_SIZE]
    lh_flat   = vec[POSE_SIZE + FACE_SIZE:
                    POSE_SIZE + FACE_SIZE + HAND_SIZE]
    rh_flat   = vec[POSE_SIZE + FACE_SIZE + HAND_SIZE:]

    pose = pose_flat.reshape(POSE_LANDMARKS, POSE_VALS)
    face = face_flat.reshape(FACE_LANDMARKS, 3)
    lh   = lh_flat.reshape(HAND_LANDMARKS, HAND_VALS)
    rh   = rh_flat.reshape(HAND_LANDMARKS, HAND_VALS)
    return pose, face, lh, rh
