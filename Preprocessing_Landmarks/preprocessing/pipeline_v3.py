from __future__ import annotations

import numpy as np

from .constants import (
    POSE_LANDMARKS, POSE_VALS,
    HAND_LANDMARKS, HAND_VALS,
    FACE_LANDMARKS, FACE_VALS,
    POSE_SIZE, HAND_SIZE, FACE_SIZE, FEATURE_DIM,
    LEG_IDXS, CRITICAL_POSE_IDXS,
)
from .geometry import in_unit_xy, reasonable_xy, valid_points_xyz
from .hands import fix_swap_and_gate_hands, fill_hand_gaps_wrist_relative_tiered
from .normalize import compute_global_root, compute_global_scale
from .smoothing import smooth_points_over_time


def preprocess_sequence_global(
    seq: np.ndarray,
    pose_vis_thresh: float = 0.1,
    keep_legs: bool = False,
    fix_swap: bool = True,
    fill_hands: bool = True,
    small_gap: int = 6,
    medium_gap: int = 15,
    min_hand_pts: int = 8,
    hand_wrist_max_dist: float = 1.1,
    rel_change_thresh: float = 0.7,
    # optional smoothing
    smooth: bool = False,
    smooth_fps: float = 20.0,
    smooth_pose: bool = True,
    smooth_hands: bool = True,
    smooth_face: bool = False,
    pose_min_cutoff: float = 1.5,
    pose_beta: float = 0.6,
    hand_min_cutoff: float = 3.0,
    hand_beta: float = 0.8,
    face_min_cutoff: float = 2.0,
    face_beta: float = 0.6,
    d_cutoff: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    v3:
    - Global-mean root+scale normalization
    - Keep wrists & other critical joints for transform using relaxed XY
    - Hand swap-fix + wrist-distance gating
    - Tiered hand gap fill
    - Optional OneEuro smoothing (adaptive anti-jitter)

    Input: (T,438)
    Output: (T,438)
    """
    y = seq.astype(np.float32, copy=True)
    if y.ndim != 2 or y.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected shape (T,{FEATURE_DIM}), got {y.shape}")

    pose = y[:, :POSE_SIZE].reshape(-1, POSE_LANDMARKS, POSE_VALS)
    face = y[:, POSE_SIZE:POSE_SIZE + FACE_SIZE].reshape(-1, FACE_LANDMARKS, FACE_VALS)
    lh   = y[:, POSE_SIZE + FACE_SIZE:POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(-1, HAND_LANDMARKS, HAND_VALS)
    rh   = y[:, POSE_SIZE + FACE_SIZE + HAND_SIZE:].reshape(-1, HAND_LANDMARKS, HAND_VALS)

    # ---- Pose cleaning: strict for visibility, relaxed for critical transform ----
    px, py, pz, pv = pose[:, :, 0], pose[:, :, 1], pose[:, :, 2], pose[:, :, 3]
    finite_pose = np.isfinite(pz) & np.isfinite(pv)

    pose_in_strict = in_unit_xy(px, py) & finite_pose
    pose_in_relaxed = reasonable_xy(px, py) & finite_pose

    critical_mask = np.zeros((pose.shape[0], POSE_LANDMARKS), dtype=bool)
    for i in CRITICAL_POSE_IDXS:
        critical_mask[:, i] = True

    # transform: allow relaxed XY for critical joints
    pose_keep_for_transform = (pv >= pose_vis_thresh) & pose_in_strict
    pose_keep_for_transform = pose_keep_for_transform | (critical_mask & pose_in_relaxed)

    # visible: keep strict
    pose_keep_visible = (pv >= pose_vis_thresh) & pose_in_strict

    bad_xyz = ~pose_keep_for_transform
    pose[bad_xyz, :3] = 0.0
    pose[~pose_keep_visible, 3] = 0.0

    if not keep_legs:
        pose[:, LEG_IDXS, :3] = 0.0
        pose[:, LEG_IDXS, 3] = 0.0

    # ---- Face cleaning (relaxed) ----
    fx, fy, fz = face[:, :, 0], face[:, :, 1], face[:, :, 2]
    face_in = reasonable_xy(fx, fy) & np.isfinite(fz)
    face[~face_in, :3] = 0.0

    # ---- Hands cleaning (relaxed) ----
    lx, ly, lz = lh[:, :, 0], lh[:, :, 1], lh[:, :, 2]
    lh_in = reasonable_xy(lx, ly) & np.isfinite(lz)
    lh[~lh_in, :3] = 0.0

    rx, ry, rz = rh[:, :, 0], rh[:, :, 1], rh[:, :, 2]
    rh_in = reasonable_xy(rx, ry) & np.isfinite(rz)
    rh[~rh_in, :3] = 0.0

    # ------------ GLOBAL ROOT + SCALE ------------
    pose_xyz = pose[:, :, :3]
    vis = pose[:, :, 3]

    root = compute_global_root(pose_xyz, vis, eps=eps)
    scale = compute_global_scale(pose_xyz, vis, root)

    pose_valid_for_transform = pose_keep_for_transform & valid_points_xyz(pose_xyz, eps=eps)
    pose_xyz[pose_valid_for_transform] = (pose_xyz[pose_valid_for_transform] - root) / scale
    pose[:, :, :3] = pose_xyz

    for arr in (face, lh, rh):
        m = valid_points_xyz(arr, eps=eps)
        arr[m] = (arr[m] - root) / scale

    # wrists in this preprocessed space
    lw = pose_xyz[:, 15, :].copy()
    rw = pose_xyz[:, 16, :].copy()

    if fix_swap:
        fix_swap_and_gate_hands(
            lh, rh, lw, rw,
            min_pts=min_hand_pts,
            hand_wrist_max_dist=hand_wrist_max_dist,
            eps=eps,
        )

    if fill_hands:
        fill_hand_gaps_wrist_relative_tiered(
            lh, lw,
            small_gap=small_gap,
            medium_gap=medium_gap,
            min_pts=min_hand_pts,
            rel_change_thresh=rel_change_thresh,
            eps=eps,
        )
        fill_hand_gaps_wrist_relative_tiered(
            rh, rw,
            small_gap=small_gap,
            medium_gap=medium_gap,
            min_pts=min_hand_pts,
            rel_change_thresh=rel_change_thresh,
            eps=eps,
        )

    # ------------ OPTIONAL SMOOTHING ------------
    if smooth:
        if smooth_pose:
            smooth_points_over_time(
                pose[:, :, :3], eps=eps, fps=smooth_fps,
                min_cutoff=pose_min_cutoff, beta=pose_beta, d_cutoff=d_cutoff
            )
        if smooth_hands:
            smooth_points_over_time(
                lh, eps=eps, fps=smooth_fps,
                min_cutoff=hand_min_cutoff, beta=hand_beta, d_cutoff=d_cutoff
            )
            smooth_points_over_time(
                rh, eps=eps, fps=smooth_fps,
                min_cutoff=hand_min_cutoff, beta=hand_beta, d_cutoff=d_cutoff
            )
        if smooth_face:
            smooth_points_over_time(
                face, eps=eps, fps=smooth_fps,
                min_cutoff=face_min_cutoff, beta=face_beta, d_cutoff=d_cutoff
            )

    out = np.empty_like(y, dtype=np.float32)
    out[:, :POSE_SIZE] = pose.reshape(-1, POSE_SIZE)
    out[:, POSE_SIZE:POSE_SIZE + FACE_SIZE] = face.reshape(-1, FACE_SIZE)
    out[:, POSE_SIZE + FACE_SIZE:POSE_SIZE + FACE_SIZE + HAND_SIZE] = lh.reshape(-1, HAND_SIZE)
    out[:, POSE_SIZE + FACE_SIZE + HAND_SIZE:] = rh.reshape(-1, HAND_SIZE)
    return out
