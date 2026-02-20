import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import json
from collections import defaultdict
import warnings
import re
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    ASL_CITIZEN_DIR = Path(r"E:\ASL_Citizen\NEW\Top_Classes_Landmarks")  # Single dataset folder
    OUTPUT_DIR = Path(r"E:\ASL_Citizen\NEW\Top_Classes_Landmarks_Preprocessed_method2")
    SPLITS_DIR = Path("./data/Enhanced_Splits_157Frames")
    ANALYSIS_DIR = Path("./data/analysis_results")
    
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Based on your analysis
    TARGET_FRAMES = 157
    FEATURE_DIM = 438
    MIN_SAMPLES_PER_WORD = 5
    
    # Frame strategy parameters
    MAX_SINGLE_FRAMES = 140
    WINDOW_THRESHOLD = 161
    VERY_LONG_THRESHOLD = 201
    
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
# GEOMETRY FUNCTIONS
# ============================================================================
def in_unit_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)

def reasonable_xy(x: np.ndarray, y: np.ndarray, lo: float = -0.25, hi: float = 1.25) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)

def valid_points_xyz(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.any(np.abs(arr) > eps, axis=-1)

def is_valid_wrist(w: np.ndarray, eps: float = 1e-8) -> bool:
    return bool(np.isfinite(w).all() and np.any(np.abs(w) > eps))

def dist2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))

# ============================================================================
# NORMALIZATION
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
# HAND FIXING
# ============================================================================
def frame_valid_hand(hand_t: np.ndarray, min_pts: int = 8, eps: float = 1e-8) -> bool:
    nz = np.any(np.abs(hand_t) > eps, axis=1)
    return int(nz.sum()) >= int(min_pts)

def hand_centroid(hand_t: np.ndarray, eps: float = 1e-8):
    m = np.any(np.abs(hand_t) > eps, axis=1)
    if not np.any(m):
        return None
    return hand_t[m].mean(axis=0)

def fix_swap_and_gate_hands(lh, rh, lw, rw, min_pts=8, hand_wrist_max_dist=1.1, eps=1e-8):
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

def fill_hand_gaps_wrist_relative_tiered(hand, wrist, small_gap=6, medium_gap=15, min_pts=8, rel_change_thresh=0.7, eps=1e-8):
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
# SMOOTHING
# ============================================================================
def _alpha(cutoff_hz: float, dt: float) -> float:
    cutoff_hz = float(max(cutoff_hz, 1e-6))
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(1.0 / (1.0 + tau / dt))

def one_euro_filter_series(x: np.ndarray, valid: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
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

def smooth_points_over_time(pts: np.ndarray, eps: float, fps: float, min_cutoff: float, beta: float, d_cutoff: float) -> None:
    T, N, _ = pts.shape
    for i in range(N):
        x = pts[:, i, :]
        valid = valid_points_xyz(x, eps=eps) & np.isfinite(x).all(axis=1)
        if not np.any(valid):
            continue
        pts[:, i, :] = one_euro_filter_series(x, valid=valid, fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

# ============================================================================
# FRAME STRATEGY (SLIDING WINDOW REMOVED)
# ============================================================================
def adaptive_padding(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    T, D = sequence.shape
    if T >= target_frames:
        return sequence[:target_frames]
    padded_seq = np.zeros((target_frames, D), dtype=np.float32)
    x_orig = np.arange(T)
    x_target = np.linspace(0, T-1, target_frames)
    for d in range(D):
        if np.any(np.isfinite(sequence[:, d])):
            if T >= 2:
                f = interpolate.interp1d(x_orig, sequence[:, d], kind='linear', bounds_error=False, fill_value="extrapolate")
                padded_seq[:, d] = f(x_target)
            else:
                padded_seq[:, d] = sequence[0, d]
    return padded_seq

def hybrid_frame_strategy(sequence: np.ndarray) -> np.ndarray:
    T = sequence.shape[0]
    if T < config.TARGET_FRAMES:
        padded = adaptive_padding(sequence, config.TARGET_FRAMES)
        return padded
    elif T > config.TARGET_FRAMES:
        return sequence[:config.TARGET_FRAMES]
    return sequence

# ============================================================================
# PREPROCESS SINGLE VIDEO
# ============================================================================
def preprocess_sequence_global(seq: np.ndarray) -> np.ndarray:
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
    fix_swap_and_gate_hands(lh, rh, lw, rw)
    fill_hand_gaps_wrist_relative_tiered(lh, lw)
    fill_hand_gaps_wrist_relative_tiered(rh, rw)

    # Smoothing
    if config.SMOOTH_POSE:
        smooth_points_over_time(pose[:, :, :3], eps=config.EPS, fps=config.FPS,
                                min_cutoff=config.POSE_MIN_CUTOFF, beta=config.POSE_BETA, d_cutoff=config.D_CUTOFF)
    if config.SMOOTH_HANDS:
        smooth_points_over_time(lh, eps=config.EPS, fps=config.FPS,
                                min_cutoff=config.HAND_MIN_CUTOFF, beta=config.HAND_BETA, d_cutoff=config.D_CUTOFF)
        smooth_points_over_time(rh, eps=config.EPS, fps=config.FPS,
                                min_cutoff=config.HAND_MIN_CUTOFF, beta=config.HAND_BETA, d_cutoff=config.D_CUTOFF)
    if config.SMOOTH_FACE:
        smooth_points_over_time(face, eps=config.EPS, fps=config.FPS,
                                min_cutoff=config.FACE_MIN_CUTOFF, beta=config.FACE_BETA, d_cutoff=config.D_CUTOFF)

    # Reconstruct
    out = np.empty_like(y, dtype=np.float32)
    out[:, :config.POSE_SIZE] = pose.reshape(-1, config.POSE_SIZE)
    out[:, config.POSE_SIZE:config.POSE_SIZE + config.FACE_SIZE] = face.reshape(-1, config.FACE_SIZE)
    out[:, config.POSE_SIZE + config.FACE_SIZE:config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE] = lh.reshape(-1, config.HAND_SIZE)
    out[:, config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE:] = rh.reshape(-1, config.HAND_SIZE)
    return out

# ============================================================================
# FILE HANDLING
# ============================================================================
def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_.]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized

def create_safe_filename(word: str, video_id: str) -> str:
    safe_word = sanitize_filename(word)
    safe_video_id = sanitize_filename(video_id)
    return f"{safe_word}_{safe_video_id}.npy"

# ============================================================================
# PROCESS SINGLE VIDEO
# ============================================================================
def process_single_video(filepath: str, word: str, video_id: str) -> str:
    raw_data = np.load(filepath, allow_pickle=True)
    cleaned_data = preprocess_sequence_global(raw_data)
    final_seq = hybrid_frame_strategy(cleaned_data)
    filename = create_safe_filename(word, video_id)
    save_path = config.OUTPUT_DIR / filename
    np.save(save_path, final_seq)
    return str(save_path)

# ============================================================================
# SCAN DATASET
# ============================================================================
def scan_dataset():
    data = []
    files = list(config.ASL_CITIZEN_DIR.glob("*.npy"))
    for f in tqdm(files, desc="Scanning ASL_CITIZEN_DIR"):
        filename = f.stem
        match = re.match(r"(.+?)\s*(\d+)$", filename)
        if match:
            word = match.group(1).lower().strip()
            video_id = match.group(2)
        else:
            word = filename.lower()
            video_id = "unknown"
        data.append({'filepath': str(f), 'word': word, 'video_id': video_id})
    return pd.DataFrame(data)

# ============================================================================
# PROCESS ALL VIDEOS
# ============================================================================
def process_all_videos():
    df = scan_dataset()
    word_counts = df['word'].value_counts()
    valid_words = word_counts[word_counts >= config.MIN_SAMPLES_PER_WORD].index
    df_filtered = df[df['word'].isin(valid_words)].copy()
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing Videos"):
        process_single_video(row['filepath'], row['word'], row['video_id'])

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*50)
    print("Enhanced Sign Language Preprocessing (No Sliding Window, No Masks)")
    print("="*50)
    process_all_videos()
    print(f"\nDone! Processed files are saved in {config.OUTPUT_DIR}")
