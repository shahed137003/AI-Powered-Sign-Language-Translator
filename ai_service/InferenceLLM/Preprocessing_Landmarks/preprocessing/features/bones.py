import numpy as np

# Pose bones (MediaPipe-style)
POSE_BONES = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (23, 24),
]

# Hand bones (chain)
HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def compute_bone_vectors(joints: np.ndarray, edges):
    bones = [joints[:, b] - joints[:, a] for a, b in edges]
    return np.concatenate(bones, axis=-1)