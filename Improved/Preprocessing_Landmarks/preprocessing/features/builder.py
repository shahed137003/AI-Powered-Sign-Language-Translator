import numpy as np

from .motion import compute_velocity, compute_acceleration
from .bones import compute_bone_vectors, POSE_BONES, HAND_BONES
from .angles import compute_hand_angles
from .masks import compute_mask, compute_frame_validity
from .temporal import temporal_encoding

from ..constants import POSE_SIZE, FACE_SIZE, HAND_SIZE


def build_features(seq: np.ndarray) -> np.ndarray:
    """
    Input: (T, 438)
    Output: (T, D)
    """

    T = seq.shape[0]

    # --------------------
    # Base
    # --------------------
    x = seq

    # --------------------
    # Motion
    # --------------------
    v = compute_velocity(x)
    #a = compute_acceleration(v)

    # --------------------
    # Structured split
    # --------------------
    pose = x[:, :POSE_SIZE].reshape(T, 33, 4)[..., :3]
    lh   = x[:, POSE_SIZE + FACE_SIZE : POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(T, 21, 3)
    rh   = x[:, POSE_SIZE + FACE_SIZE + HAND_SIZE :].reshape(T, 21, 3)

    # --------------------
    # Bones
    # --------------------
    pose_bones = compute_bone_vectors(pose, POSE_BONES)
    lh_bones   = compute_bone_vectors(lh, HAND_BONES)
    rh_bones   = compute_bone_vectors(rh, HAND_BONES)

    # --------------------
    # Angles
    # --------------------
    #lh_angles = compute_hand_angles(lh)
    #rh_angles = compute_hand_angles(rh)

    # --------------------
    # Masks
    # --------------------
    # mask = compute_mask(x)
    # frame_valid = compute_frame_validity(x)

    # --------------------
    # Temporal
    # --------------------
    # t_enc = temporal_encoding(T)

    # --------------------
    # FINAL CONCAT
    # --------------------
    features = np.concatenate([
    x,
    v,
    pose_bones,
    lh_bones,
    rh_bones,
], axis=-1)

    return features.astype(np.float32)