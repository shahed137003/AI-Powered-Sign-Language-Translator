# ----------------------------
# Layout (pose, face, LH, RH) - 438 schema
# ----------------------------
POSE_LANDMARKS, POSE_VALS = 33, 4
HAND_LANDMARKS, HAND_VALS = 21, 3
FACE_LANDMARKS, FACE_VALS = 60, 3

POSE_SIZE = POSE_LANDMARKS * POSE_VALS          # 132
HAND_SIZE = HAND_LANDMARKS * HAND_VALS          # 63
FACE_SIZE = FACE_LANDMARKS * FACE_VALS          # 180
FEATURE_DIM = POSE_SIZE + FACE_SIZE + 2 * HAND_SIZE  # 438

# Legs to drop (knees->feet). Hips (23,24) are kept.
LEG_IDXS = list(range(25, 33))  # 25..32 inclusive

# Critical joints we keep for transform even if visibility is low
CRITICAL_POSE_IDXS = {0, 11, 12, 13, 14, 15, 16, 23, 24}  # nose, shoulders, elbows, wrists, hips
