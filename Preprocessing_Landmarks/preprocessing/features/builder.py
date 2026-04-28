import numpy as np

from .bones import compute_bone_vectors, POSE_BONES, HAND_BONES
from .angles import compute_hand_angles
from ..constants import POSE_SIZE, FACE_SIZE, HAND_SIZE


def build_features(seq: np.ndarray) -> np.ndarray:
    T = seq.shape[0]
    x = seq

    valid = (np.abs(x).sum(axis=-1) > 0)

    v = np.zeros_like(x)
    a = np.zeros_like(x)

    for t in range(1, T):
        if valid[t] and valid[t - 1]:
            v[t] = x[t] - x[t - 1]
        else:
            v[t] = 0.0

    for t in range(1, T):
        if valid[t] and valid[t - 1]:
            a[t] = v[t] - v[t - 1]
        else:
            a[t] = 0.0

    # smoother than hard clipping
    a = np.tanh(a)

    pose = x[:, :POSE_SIZE].reshape(T, 33, 4)[..., :3]
    lh   = x[:, POSE_SIZE + FACE_SIZE : POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(T, 21, 3)
    rh   = x[:, POSE_SIZE + FACE_SIZE + HAND_SIZE :].reshape(T, 21, 3)

    pose_bones = compute_bone_vectors(pose, POSE_BONES)
    lh_bones   = compute_bone_vectors(lh, HAND_BONES)
    rh_bones   = compute_bone_vectors(rh, HAND_BONES)

    lh_angles = compute_hand_angles(lh)
    rh_angles = compute_hand_angles(rh)

    features = np.concatenate([
        x,
        v,
        a,
        pose_bones,
        lh_bones,
        rh_bones,
        lh_angles,
        rh_angles,
    ], axis=-1)

    return features.astype(np.float32)