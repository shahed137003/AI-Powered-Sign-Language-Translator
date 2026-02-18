import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy import interpolate  # <-- Missing import for adaptive_padding function
from collections import defaultdict
import json
from IPython.display import HTML
# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    MSASL_RAW_DIR = Path(r"E:\ASL_Citizen\NEW\Top_Classes_Landmarks")  # <-- only one dataset
    WLASL_RAW_DIR = None  # or just ignore WLASL usage
    OUTPUT_DIR = Path(r"E:\ASL_Citizen\NEW\Top_Classes_Landmarks_Preprocessed")
    SPLITS_DIR = Path("./data/Enhanced_Splits_157Frames")
    ANALYSIS_DIR = Path("./data/analysis_results")
    
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Based on your analysis
    TARGET_FRAMES = 157                    # <-- changed
    FEATURE_DIM = 438                     # same as before
    MIN_SAMPLES_PER_WORD = 5              # same as before
    
    # Frame strategy parameters (from your analysis)
    MAX_SINGLE_FRAMES = 140               # same
    WINDOW_THRESHOLD = 161                # same
    VERY_LONG_THRESHOLD = 201             # same
    
    # Geometry constants
    POSE_SIZE = 132
    HAND_SIZE = 63
    FACE_SIZE = 180
    POSE_LANDMARKS, POSE_VALS = 33, 4
    HAND_LANDMARKS, HAND_VALS = 21, 3
    FACE_LANDMARKS, FACE_VALS = 60, 3
    LEG_IDXS = list(range(25, 33))
    CRITICAL_POSE_IDXS = {0, 11, 12, 13, 14, 15, 16, 23, 24}
    
    # Preprocessing parameters
    SMOOTH_POSE = True
    SMOOTH_HANDS = True
    SMOOTH_FACE = False
    POSE_MIN_CUTOFF = 1.5
    POSE_BETA = 0.4
    HAND_MIN_CUTOFF = 2.0
    HAND_BETA = 0.3
    FACE_MIN_CUTOFF = 2.0
    FACE_BETA = 0.4
    D_CUTOFF = 1.0
    FPS = 20.0
    EPS = 1e-8

config = Config()
# ============================================================================
# 1. ENHANCED GEOMETRY FUNCTIONS (Keep your original)
# ============================================================================
def in_unit_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask where x,y are finite and within [0,1]."""
    return (
        np.isfinite(x) & np.isfinite(y) &
        (x >= 0.0) & (x <= 1.0) &
        (y >= 0.0) & (y <= 1.0)
    )

def reasonable_xy(x: np.ndarray, y: np.ndarray, lo: float = -0.25, hi: float = 1.25) -> np.ndarray:
    """Relaxed normalized check (helps during fast motion / partial crops)."""
    return (
        np.isfinite(x) & np.isfinite(y) &
        (x >= lo) & (x <= hi) &
        (y >= lo) & (y <= hi)
    )

def valid_points_xyz(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """arr shape (...,3) -> mask over last axis: point is not all-zero."""
    return np.any(np.abs(arr) > eps, axis=-1)

def is_valid_wrist(w: np.ndarray, eps: float = 1e-8) -> bool:
    return bool(np.isfinite(w).all() and np.any(np.abs(w) > eps))

def dist2(a: np.ndarray, b: np.ndarray) -> float:
    """2D distance in XY."""
    return float(np.linalg.norm(a[:2] - b[:2]))
# ============================================================================
# 2. ENHANCED NORMALIZATION (Keep your original)
# ============================================================================
def compute_global_root(pose_xyz: np.ndarray, vis: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    def collect_mid(i1: int, i2: int):
        m = (vis[:, i1] > 0.0) & (vis[:, i2] > 0.0)
        m = m & valid_points_xyz(pose_xyz[:, i1, :], eps) & valid_points_xyz(pose_xyz[:, i2, :], eps)
        if not np.any(m):
            return None
        return (pose_xyz[m, i1, :] + pose_xyz[m, i2, :]) / 2.0

    mid_hip = collect_mid(23, 24)
    if mid_hip is not None:
        return mid_hip.mean(axis=0)

    mid_sh = collect_mid(11, 12)
    if mid_sh is not None:
        return mid_sh.mean(axis=0)

    m_nose = (vis[:, 0] > 0.0) & valid_points_xyz(pose_xyz[:, 0, :], eps)
    if np.any(m_nose):
        return pose_xyz[m_nose, 0, :].mean(axis=0)

    m_all = (vis > 0.0) & valid_points_xyz(pose_xyz, eps)
    if np.any(m_all):
        return pose_xyz[m_all].mean(axis=0)

    return np.zeros(3, dtype=np.float32)

def compute_global_scale(pose_xyz: np.ndarray, vis: np.ndarray, root: np.ndarray, eps: float = 1e-6) -> float:
    def collect_dist(i1: int, i2: int):
        m = (vis[:, i1] > 0.0) & (vis[:, i2] > 0.0)
        m = m & valid_points_xyz(pose_xyz[:, i1, :]) & valid_points_xyz(pose_xyz[:, i2, :])
        if not np.any(m):
            return None
        d = np.linalg.norm(pose_xyz[m, i1, :] - pose_xyz[m, i2, :], axis=1)
        d = d[d > eps]
        return d if d.size > 0 else None

    d_sh = collect_dist(11, 12)
    if d_sh is not None:
        return float(d_sh.mean())

    d_hip = collect_dist(23, 24)
    if d_hip is not None:
        return float(d_hip.mean())

    m_all = (vis > 0.0) & valid_points_xyz(pose_xyz)
    if np.any(m_all):
        d = np.linalg.norm(pose_xyz[m_all] - root[None, :], axis=1)
        d = d[d > eps]
        if d.size > 0:
            return float(d.mean())

    return 1.0
# ============================================================================
# 3. ENHANCED HAND FIXING (Keep your original)
# ============================================================================
def frame_valid_hand(hand_t: np.ndarray, min_pts: int = 8, eps: float = 1e-8) -> bool:
    """A frame counts as 'hand present' if it has >= min_pts non-zero landmarks."""
    nz = np.any(np.abs(hand_t) > eps, axis=1)
    return int(nz.sum()) >= int(min_pts)

def hand_centroid(hand_t: np.ndarray, eps: float = 1e-8):
    m = np.any(np.abs(hand_t) > eps, axis=1)
    if not np.any(m):
        return None
    return hand_t[m].mean(axis=0)

def fix_swap_and_gate_hands(
    lh: np.ndarray, rh: np.ndarray,
    lw: np.ndarray, rw: np.ndarray,
    min_pts: int = 8,
    hand_wrist_max_dist: float = 1.1,
    eps: float = 1e-8,
) -> None:
    T = lh.shape[0]
    for t in range(T):
        l_ok = frame_valid_hand(lh[t], min_pts=min_pts, eps=eps)
        r_ok = frame_valid_hand(rh[t], min_pts=min_pts, eps=eps)

        wl_ok = is_valid_wrist(lw[t], eps=eps)
        wr_ok = is_valid_wrist(rw[t], eps=eps)

        cL = hand_centroid(lh[t], eps=eps) if l_ok else None
        cR = hand_centroid(rh[t], eps=eps) if r_ok else None

        if l_ok and r_ok and wl_ok and wr_ok and (cL is not None) and (cR is not None):
            d_ll = dist2(cL, lw[t])
            d_lr = dist2(cL, rw[t])
            d_rr = dist2(cR, rw[t])
            d_rl = dist2(cR, lw[t])
            if (d_lr + d_rl) + 1e-6 < (d_ll + d_rr):
                lh[t], rh[t] = rh[t].copy(), lh[t].copy()
                cL, cR = cR, cL

        if wl_ok and l_ok and (cL is not None):
            if dist2(cL, lw[t]) > hand_wrist_max_dist:
                lh[t] = 0.0
        if wr_ok and r_ok and (cR is not None):
            if dist2(cR, rw[t]) > hand_wrist_max_dist:
                rh[t] = 0.0

def fill_hand_gaps_wrist_relative_tiered(
    hand: np.ndarray,
    wrist: np.ndarray,
    small_gap: int = 6,
    medium_gap: int = 15,
    min_pts: int = 8,
    rel_change_thresh: float = 0.7,
    eps: float = 1e-8,
) -> None:
    T = hand.shape[0]
    valid = np.array([frame_valid_hand(hand[t], min_pts=min_pts, eps=eps) for t in range(T)], dtype=bool)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return

    def set_from_rel(t: int, rel: np.ndarray):
        if is_valid_wrist(wrist[t], eps=eps):
            hand[t] = rel + wrist[t]

    for a, b in zip(idx[:-1], idx[1:]):
        gap = int(b - a - 1)
        if gap <= 0:
            continue
        if gap > medium_gap:
            continue

        if not (is_valid_wrist(wrist[a], eps=eps) and is_valid_wrist(wrist[b], eps=eps)):
            if gap <= small_gap:
                for t in range(a + 1, b):
                    hand[t] = hand[a]
            continue

        rel_a = hand[a] - wrist[a]
        rel_b = hand[b] - wrist[b]

        if gap > small_gap:
            for t in range(a + 1, b):
                set_from_rel(t, rel_a)
            continue

        delta = np.linalg.norm(rel_a - rel_b, axis=1)
        delta = delta[np.isfinite(delta)]
        rel_delta = float(np.median(delta)) if delta.size else 999.0

        if rel_delta <= rel_change_thresh:
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                rel = (1.0 - alpha) * rel_a + alpha * rel_b
                set_from_rel(t, rel)
        else:
            for t in range(a + 1, b):
                set_from_rel(t, rel_a)
# ============================================================================
# 4. ENHANCED SMOOTHING WITH SELECTIVE APPLICATION
# ============================================================================
def _alpha(cutoff_hz: float, dt: float) -> float:
    cutoff_hz = float(max(cutoff_hz, 1e-6))
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(1.0 / (1.0 + tau / dt))

def one_euro_filter_series(
    x: np.ndarray,
    valid: np.ndarray,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    T, D = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    dt = 1.0 / float(max(fps, 1e-6))

    x_prev = np.zeros(D, dtype=np.float32)
    x_hat_prev = np.zeros(D, dtype=np.float32)
    dx_hat_prev = np.zeros(D, dtype=np.float32)
    has_prev = False

    for t in range(T):
        if not bool(valid[t]):
            has_prev = False
            continue

        xt = x[t].astype(np.float32, copy=False)

        if not has_prev:
            out[t] = xt
            x_prev = xt
            x_hat_prev = xt
            dx_hat_prev[:] = 0.0
            has_prev = True
            continue

        dx = (xt - x_prev) / dt
        a_d = _alpha(d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat_prev

        cutoff = float(min_cutoff + beta * np.linalg.norm(dx_hat))
        a = _alpha(cutoff, dt)

        x_hat = a * xt + (1.0 - a) * x_hat_prev

        out[t] = x_hat
        x_prev = xt
        x_hat_prev = x_hat
        dx_hat_prev = dx_hat

    return out

def smooth_points_over_time(
    pts: np.ndarray,
    eps: float,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> None:
    T, N, _ = pts.shape
    for i in range(N):
        x = pts[:, i, :]
        valid = valid_points_xyz(x, eps=eps) & np.isfinite(x).all(axis=1)
        if not np.any(valid):
            continue
        pts[:, i, :] = one_euro_filter_series(
            x, valid=valid, fps=fps,
            min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
        )
# ============================================================================
# 5. FRAME MANAGEMENT - HYBRID STRATEGY
# ============================================================================
def adaptive_padding(sequence: np.ndarray, target_frames: int) -> tuple:
    """
    Smart padding for short sequences using interpolation.
    Returns: (padded_sequence, attention_mask)
    """
    T, D = sequence.shape
    
    if T >= target_frames:
        return sequence[:target_frames], np.ones(target_frames, dtype=np.float32)
    
    # Create padded sequence with interpolation
    padded_seq = np.zeros((target_frames, D), dtype=np.float32)
    
    # Original time points
    x_orig = np.arange(T)
    x_target = np.linspace(0, T-1, target_frames)
    
    # Interpolate each feature dimension
    for d in range(D):
        if np.any(np.isfinite(sequence[:, d])):
            # Use linear interpolation
            if T >= 2:
                f = interpolate.interp1d(
                    x_orig, sequence[:, d], 
                    kind='linear',
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                padded_seq[:, d] = f(x_target)
            else:
                # If only 1 frame, repeat it
                padded_seq[:, d] = sequence[0, d]
    
    # Create attention mask (1 for interpolated frames close to original, 0.5 for extended)
    mask = np.zeros(target_frames, dtype=np.float32)
    
    # Mark positions corresponding to original frames
    for i, target_pos in enumerate(x_target):
        closest_orig = int(round(target_pos))
        if 0 <= closest_orig < T:
            mask[i] = 1.0
        else:
            mask[i] = 0.5  # Extended frames
    
    return padded_seq, mask

def create_sliding_windows(sequence: np.ndarray, window_size: int, overlap: float = 0.5) -> list:
    """
    Create sliding windows for long sequences.
    """
    T = sequence.shape[0]
    windows = []
    
    if T <= window_size:
        return [sequence]
    
    # Calculate step size
    step = int(window_size * (1 - overlap))
    if step < 1:
        step = 1
    
    # Create windows
    start = 0
    while start + window_size <= T:
        windows.append(sequence[start:start + window_size])
        start += step
    
    # Always include the last window
    if not windows or (start < T and T >= window_size):
        last_window = sequence[-window_size:]
        if len(windows) == 0 or not np.array_equal(last_window, windows[-1]):
            windows.append(last_window)
    
    return windows

def hybrid_frame_strategy(sequence: np.ndarray, original_length: int) -> tuple:
    """
    FIXED VERSION: Hybrid frame strategy based on sequence length.
    Returns (sequences_list, masks_list, metadata_list) where masks have shape (T,)
    """
    T = sequence.shape[0]  # FIXED: Use actual sequence length, not original_length
    
    # Debug: Print actual vs original
    if T != original_length:
        print(f"  Note: Sequence length changed from {original_length} to {T}")
    
    if T < 30:
        # Case 1: Very Short - Interpolation padding
        padded_seq, mask = adaptive_padding(sequence, config.TARGET_FRAMES)
        return [padded_seq], [mask], [{
            'strategy': 'interpolation_padding',
            'original_length': original_length,
            'processed_length': T,
            'padding_amount': config.TARGET_FRAMES - T
        }]
    
    elif T < config.TARGET_FRAMES:
        # Case 2: Short - Edge padding (repeat last frame)
        padded_seq = np.zeros((config.TARGET_FRAMES, sequence.shape[1]), dtype=np.float32)
        padded_seq[:T] = sequence
        
        if T > 0:
            padded_seq[T:] = sequence[-1]
        
        mask = np.zeros(config.TARGET_FRAMES, dtype=np.float32)
        mask[:T] = 1.0
        
        return [padded_seq], [mask], [{
            'strategy': 'edge_padding',
            'original_length': original_length,
            'processed_length': T,
            'padding_amount': config.TARGET_FRAMES - T
        }]
    
    elif T <= config.MAX_SINGLE_FRAMES:
        # Case 3: Medium - Take first 96 frames
        truncated_seq = sequence[:config.TARGET_FRAMES]
        mask = np.ones(config.TARGET_FRAMES, dtype=np.float32)
        
        return [truncated_seq], [mask], [{
            'strategy': 'first_frames',
            'original_length': original_length,
            'processed_length': T,
            'truncation_amount': T - config.TARGET_FRAMES
        }]
    
    elif T <= config.WINDOW_THRESHOLD:
        # Case 4: Long - Take middle 96 frames
        middle_start = (T - config.TARGET_FRAMES) // 2
        middle_seq = sequence[middle_start:middle_start + config.TARGET_FRAMES]
        mask = np.ones(config.TARGET_FRAMES, dtype=np.float32)
        
        return [middle_seq], [mask], [{
            'strategy': 'middle_frames',
            'original_length': original_length,
            'processed_length': T,
            'middle_start': middle_start
        }]
    
    else:
        # Case 5: Very Long - Sliding windows
        windows = create_sliding_windows(sequence, config.TARGET_FRAMES, overlap=0.5)
        masks = [np.ones(config.TARGET_FRAMES, dtype=np.float32) for _ in windows]
        metadata = []
        
        for i, window in enumerate(windows):
            metadata.append({
                'strategy': 'sliding_window',
                'original_length': original_length,
                'processed_length': T,
                'window_index': i,
                'total_windows': len(windows),
                'overlap_ratio': 0.5
            })
        
        return windows, masks, metadata
# ============================================================================
# 6. MAIN PREPROCESSING PIPELINE
# ============================================================================
def preprocess_sequence_global(seq: np.ndarray) -> np.ndarray:
    """
    Core preprocessing pipeline (your original function).
    Returns cleaned sequence.
    """
    y = seq.astype(np.float32, copy=True)
    if y.ndim != 2 or y.shape[1] != config.FEATURE_DIM:
        raise ValueError(f"Expected shape (T,{config.FEATURE_DIM}), got {y.shape}")

    pose = y[:, :config.POSE_SIZE].reshape(-1, config.POSE_LANDMARKS, config.POSE_VALS)
    face = y[:, config.POSE_SIZE:config.POSE_SIZE + config.FACE_SIZE].reshape(-1, config.FACE_LANDMARKS, config.FACE_VALS)
    lh = y[:, config.POSE_SIZE + config.FACE_SIZE:config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE].reshape(-1, config.HAND_LANDMARKS, config.HAND_VALS)
    rh = y[:, config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE:].reshape(-1, config.HAND_LANDMARKS, config.HAND_VALS)

    # Pose cleaning
    px, py, pz, pv = pose[:, :, 0], pose[:, :, 1], pose[:, :, 2], pose[:, :, 3]
    finite_pose = np.isfinite(pz) & np.isfinite(pv)
    
    pose_in_strict = in_unit_xy(px, py) & finite_pose
    pose_in_relaxed = reasonable_xy(px, py) & finite_pose
    
    critical_mask = np.zeros((pose.shape[0], config.POSE_LANDMARKS), dtype=bool)
    for i in config.CRITICAL_POSE_IDXS:
        critical_mask[:, i] = True
    
    pose_keep_for_transform = (pv >= 0.1) & pose_in_strict
    pose_keep_for_transform = pose_keep_for_transform | (critical_mask & pose_in_relaxed)
    
    pose_keep_visible = (pv >= 0.1) & pose_in_strict
    
    bad_xyz = ~pose_keep_for_transform
    pose[bad_xyz, :3] = 0.0
    pose[~pose_keep_visible, 3] = 0.0
    
    pose[:, config.LEG_IDXS, :3] = 0.0
    pose[:, config.LEG_IDXS, 3] = 0.0
    
    # Face cleaning
    fx, fy, fz = face[:, :, 0], face[:, :, 1], face[:, :, 2]
    face_in = reasonable_xy(fx, fy) & np.isfinite(fz)
    face[~face_in, :3] = 0.0
    
    # Hands cleaning
    lx, ly, lz = lh[:, :, 0], lh[:, :, 1], lh[:, :, 2]
    lh_in = reasonable_xy(lx, ly) & np.isfinite(lz)
    lh[~lh_in, :3] = 0.0
    
    rx, ry, rz = rh[:, :, 0], rh[:, :, 1], rh[:, :, 2]
    rh_in = reasonable_xy(rx, ry) & np.isfinite(rz)
    rh[~rh_in, :3] = 0.0
    
    # Global normalization
    pose_xyz = pose[:, :, :3]
    vis = pose[:, :, 3]
    
    root = compute_global_root(pose_xyz, vis, eps=config.EPS)
    scale = compute_global_scale(pose_xyz, vis, root)
    
    pose_valid_for_transform = pose_keep_for_transform & valid_points_xyz(pose_xyz, eps=config.EPS)
    pose_xyz[pose_valid_for_transform] = (pose_xyz[pose_valid_for_transform] - root) / scale
    pose[:, :, :3] = pose_xyz
    
    for arr in (face, lh, rh):
        m = valid_points_xyz(arr, eps=config.EPS)
        arr[m] = (arr[m] - root) / scale
    
    # Wrist positions
    lw = pose_xyz[:, 15, :].copy()
    rw = pose_xyz[:, 16, :].copy()
    
    # Hand fixing
    fix_swap_and_gate_hands(
        lh, rh, lw, rw,
        min_pts=8,
        hand_wrist_max_dist=1.1,
        eps=config.EPS,
    )
    
    fill_hand_gaps_wrist_relative_tiered(
        lh, lw,
        small_gap=6,
        medium_gap=15,
        min_pts=8,
        rel_change_thresh=0.7,
        eps=config.EPS,
    )
    fill_hand_gaps_wrist_relative_tiered(
        rh, rw,
        small_gap=6,
        medium_gap=15,
        min_pts=8,
        rel_change_thresh=0.7,
        eps=config.EPS,
    )
    
    # Smoothing
    if config.SMOOTH_POSE:
        smooth_points_over_time(
            pose[:, :, :3], eps=config.EPS, fps=config.FPS,
            min_cutoff=config.POSE_MIN_CUTOFF, beta=config.POSE_BETA, d_cutoff=config.D_CUTOFF
        )
    if config.SMOOTH_HANDS:
        smooth_points_over_time(
            lh, eps=config.EPS, fps=config.FPS,
            min_cutoff=config.HAND_MIN_CUTOFF, beta=config.HAND_BETA, d_cutoff=config.D_CUTOFF
        )
        smooth_points_over_time(
            rh, eps=config.EPS, fps=config.FPS,
            min_cutoff=config.HAND_MIN_CUTOFF, beta=config.HAND_BETA, d_cutoff=config.D_CUTOFF
        )
    if config.SMOOTH_FACE:
        smooth_points_over_time(
            face, eps=config.EPS, fps=config.FPS,
            min_cutoff=config.FACE_MIN_CUTOFF, beta=config.FACE_BETA, d_cutoff=config.D_CUTOFF
        )
    
    # Reconstruct
    out = np.empty_like(y, dtype=np.float32)
    out[:, :config.POSE_SIZE] = pose.reshape(-1, config.POSE_SIZE)
    out[:, config.POSE_SIZE:config.POSE_SIZE + config.FACE_SIZE] = face.reshape(-1, config.FACE_SIZE)
    out[:, config.POSE_SIZE + config.FACE_SIZE:config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE] = lh.reshape(-1, config.HAND_SIZE)
    out[:, config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE:] = rh.reshape(-1, config.HAND_SIZE)
    
    return out
# ============================================================================
# FIXED: FILENAME HANDLING WITH SAFE CHARACTERS
# ============================================================================
import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing special characters.
    Keeps only alphanumeric, underscores, and dots.
    """
    # Remove special characters but keep alphanumeric, underscore, dot
    sanitized = re.sub(r'[^a-zA-Z0-9_.]', '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def create_safe_filename(source: str, word: str, video_id: str, window_idx: int = None) -> tuple:
    """
    Create safe filenames for processed files and masks.
    source parameter is ignored (removed from filename)
    """
    safe_word = sanitize_filename(word)
    safe_video_id = sanitize_filename(video_id)
    
    # Base filename: only word + video_id
    if window_idx is None:
        base_name = f"{safe_word}_{safe_video_id}"
    else:
        base_name = f"{safe_word}_{safe_video_id}_w{window_idx:02d}"
    
    # Full filenames
    processed_filename = f"{base_name}.npy"
    mask_filename = f"{base_name}_mask.npy"
    
    return base_name, processed_filename, mask_filename


# ============================================================================
# UPDATED: PROCESS SINGLE VIDEO FUNCTION
# ============================================================================
def process_single_video_fixed(filepath: str, word: str, source: str, video_id: str) -> list:
    """
    Process a single video file with safe filename handling.
    Returns list of processed records.
    """
    records = []
    
    try:
        # Load raw data
        raw_data = np.load(filepath, allow_pickle=True)
        original_length = raw_data.shape[0]
        
        # Apply core preprocessing
        cleaned_data = preprocess_sequence_global(raw_data)
        
        # Debug: Check shape
        print(f"  Original: {raw_data.shape}, Cleaned: {cleaned_data.shape}")
        
        # Apply hybrid frame strategy
        sequences, masks, metadata = hybrid_frame_strategy(cleaned_data, original_length)
        
        # DEBUG: Show what hybrid strategy produced
        print(f"  → Strategy produced {len(sequences)} sequence(s)")
        for i, (seq, mask) in enumerate(zip(sequences, masks)):
            print(f"    Seq {i+1}: shape={seq.shape}, mask={mask.shape}")
            print(f"    Strategy: {metadata[i]['strategy']}")
        
        # VERIFY SHAPES BEFORE SAVING
        for seq_idx, (seq, mask, meta) in enumerate(zip(sequences, masks, metadata)):
            # CRITICAL: Check if shapes are correct
            if seq.shape != (config.TARGET_FRAMES, config.FEATURE_DIM):
                print(f"   WARNING: Sequence shape {seq.shape}, expected (96, 438)!")
                # Fix it
                if seq.shape[0] < config.TARGET_FRAMES:
                    # Pad with zeros
                    padded = np.zeros((config.TARGET_FRAMES, config.FEATURE_DIM))
                    padded[:seq.shape[0]] = seq
                    seq = padded
                    print(f"  → Padded from {seq.shape[0]} to {config.TARGET_FRAMES} frames")
                else:
                    # Truncate
                    seq = seq[:config.TARGET_FRAMES]
                    print(f"  → Truncated from {seq.shape[0]} to {config.TARGET_FRAMES} frames")
            
            if mask.shape != (config.TARGET_FRAMES,):
                print(f"  WARNING: Mask shape {mask.shape}, expected (96,)!")
                # Fix it
                if mask.shape[0] < config.TARGET_FRAMES:
                    padded_mask = np.zeros(config.TARGET_FRAMES)
                    padded_mask[:mask.shape[0]] = mask
                    mask = padded_mask
                else:
                    mask = mask[:config.TARGET_FRAMES]
            
            # Generate safe filenames
            window_idx = None if len(sequences) == 1 else seq_idx
            base_name, processed_filename, mask_filename = create_safe_filename(
                source, word, video_id, window_idx
            )
            
            save_path = config.OUTPUT_DIR / processed_filename
            mask_path = config.OUTPUT_DIR / mask_filename
            
            # Save data
            np.save(save_path, seq)
            np.save(mask_path, mask)
            
            # Create record
            records.append({
                'word': word,
                'original_file': filepath,
                'processed_file': str(save_path),
                'mask_file': str(mask_path),
                'source': source,
                'video_id': video_id,
                'original_length': original_length,
                'processed_length': seq.shape[0],
                'strategy': meta['strategy'],
                'is_windowed': meta['strategy'] == 'sliding_window',
                'window_index': meta.get('window_index', 0),
                'total_windows': meta.get('total_windows', 1),
                'padding_amount': meta.get('padding_amount', 0),
                'truncation_amount': meta.get('truncation_amount', 0),
                'base_filename': base_name
            })
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return records


# ============================================================================
# UPDATED: SCAN DATASETS FUNCTION WITH BETTER ERROR HANDLING
# ============================================================================
import re

def scan_datasets():
    """Scan only MSASL_RAW_DIR (your single folder) and fix word extraction."""
    print("Scanning dataset...")

    def scan_dir(path, source_name):
        data = []
        if not path.exists():
            print(f" {source_name} directory not found: {path}")
            return data

        files = list(path.glob("*.npy"))
        print(f"Found {len(files)} .npy files in {source_name}")

        for f in tqdm(files, desc=f"Scanning {source_name}"):
            try:
                filename = f.stem  # "ABOUT 5"
                
                # Split word and numeric ID at the end
                match = re.match(r"(.+?)\s*(\d+)$", filename)
                if match:
                    word = match.group(1)
                    video_id = match.group(2)
                else:
                    word = filename
                    video_id = "unknown"
                
                word = word.lower().strip()
                
                data.append({
                    'filepath': str(f),
                    'word': word,
                    'source': source_name,
                    'video_id': video_id,
                    'original_filename': filename
                })
            except Exception as e:
                print(f"   Error parsing {f.name}: {e}")
                continue
        return data

    msasl_data = scan_dir(config.MSASL_RAW_DIR, "TOP_CLASSES")
    df = pd.DataFrame(msasl_data)
    return df


# ============================================================================
# 10. SIMPLIFIED PROCESSING (Minimal Prints)
# ============================================================================
def process_all_videos_silent():
    """Main processing pipeline with minimal output."""
    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scan datasets
    df = scan_datasets()
    if df.empty:
        print(" No files found!")
        return None
    
    print(f"Found {len(df):,} videos")
    
    # Filter rare words
    word_counts = df['word'].value_counts()
    valid_words = word_counts[word_counts >= config.MIN_SAMPLES_PER_WORD].index
    df_filtered = df[df['word'].isin(valid_words)].copy()
    
    print(f"Processing {len(df_filtered):,} videos after filtering...")
    
    # Process videos
    all_records = []
    stats = defaultdict(int)
    
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing"):
        # FIXED: Call the renamed function
        records = process_single_video_fixed(
            row['filepath'],
            row['word'],
            row['source'],
            row['video_id']
        )
        
        if records:
            all_records.extend(records)
            for record in records:
                stats[record['strategy']] += 1
                stats['total'] += 1
    
    if not all_records:
        print("No videos processed!")
        return None
    
    # Save metadata
    df_processed = pd.DataFrame(all_records)
    df_processed.to_csv(config.OUTPUT_DIR / "metadata.csv", index=False)
    
    # Print quick summary
    print(f"\nDone: {stats['total']:,} samples generated")
    for strategy in ['interpolation_padding', 'edge_padding', 'first_frames', 
                     'middle_frames', 'sliding_window']:
        if strategy in stats:
            print(f"  {strategy}: {stats[strategy]:,}")
    
    return df_processed
def test_few_files():
    """Test the fixed code on 3 files to see what hybrid_frame_strategy produces."""
    print(" Testing hybrid_frame_strategy on sample files...")
    print("-" * 60)
    
    # Find a few files
    test_files = list(config.MSASL_RAW_DIR.glob("*.npy"))[:3]
    
    for filepath in test_files:
        print(f"\nFile: {filepath.name}")
        try:
            raw_data = np.load(filepath, allow_pickle=True)
            print(f"  Original: {raw_data.shape}")
            
            cleaned = preprocess_sequence_global(raw_data)
            print(f"  Cleaned: {cleaned.shape}")
            
            sequences, masks, metadata = hybrid_frame_strategy(cleaned, raw_data.shape[0])
            
            print(f"  Strategy: {metadata[0]['strategy']}")
            print(f"  Produced {len(sequences)} sequence(s):")
            
            for i, (seq, mask) in enumerate(zip(sequences, masks)):
                print(f"    Sequence {i+1}: {seq.shape}")
                print(f"    Mask {i+1}: {mask.shape}")
                
                # Check if it's 96 frames
                if seq.shape == (config.TARGET_FRAMES, config.FEATURE_DIM):
                    print(f"Correct shape!")
                else:
                    print(f"Wrong shape! Expected (96, 438)")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    print("-" * 60)
# ============================================================================
# 14. CLEAN MAIN EXECUTION
# ============================================================================
def main_clean():
    """Clean main function with minimal output."""
    print("=" * 50)
    print("Enhanced Sign Language Preprocessing")
    print("=" * 50)
    
    # 1. Process videos
    df_processed = process_all_videos_silent()
    
    if df_processed is not None:
        # 2. Create splits
        df_train, df_val, df_test, label_map = create_quick_splits(df_processed)
        
        # 3. Quick comparison with original
        print("\n Sample Check:")
        sample = df_processed.iloc[0]
        try:
            data = np.load(sample['processed_file'])
            mask = np.load(sample['mask_file'])
            print(f"  Sample shape: {data.shape} (mask: {mask.shape})")
            print(f"  Strategy: {sample['strategy']}")
        except:
            print("  Could not load sample file")
        
        print(f"\nDone! Check {config.OUTPUT_DIR}/ for files.")

if __name__ == "__main__":
    main_clean()

