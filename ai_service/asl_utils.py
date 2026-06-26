import numpy as np
import cv2
from scipy import interpolate

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
class Config:
    TARGET_FRAMES = 157
    FEATURE_DIM = 438
    MAX_SINGLE_FRAMES = 140
    WINDOW_THRESHOLD = 161
    
    POSE_SIZE, HAND_SIZE, FACE_SIZE = 132, 63, 180
    POSE_LANDMARKS, POSE_VALS = 33, 4
    HAND_LANDMARKS, HAND_VALS = 21, 3
    FACE_LANDMARKS, FACE_VALS = 60, 3
    
    LEG_IDXS = list(range(25, 33))
    CRITICAL_POSE_IDXS = {0, 11, 12, 13, 14, 15, 16, 23, 24}
    
    SMOOTH_POSE, SMOOTH_HANDS, SMOOTH_FACE = True, True, False
    POSE_MIN_CUTOFF, POSE_BETA = 1.5, 0.4
    HAND_MIN_CUTOFF, HAND_BETA = 2.0, 0.3
    FACE_MIN_CUTOFF, FACE_BETA = 2.0, 0.4
    D_CUTOFF, FPS, EPS = 1.0, 20.0, 1e-8

config = Config()

# ============================================================================
# MATH & GEOMETRY HELPER FUNCTIONS
# ============================================================================
def in_unit_xy(x, y): 
    return np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
def reasonable_xy(x, y, lo=-0.25, hi=1.25): 
    return np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
def valid_points_xyz(arr, eps=1e-8): 
    return np.any(np.abs(arr) > eps, axis=-1)
def is_valid_wrist(w, eps=1e-8): 
    return bool(np.isfinite(w).all() and np.any(np.abs(w) > eps))
def dist2(a, b): 
    return float(np.linalg.norm(a[:2] - b[:2]))

def compute_global_root(pose_xyz, vis, eps=1e-8):
    def collect_mid(i1, i2):
        m = (vis[:, i1] > 0.0) & (vis[:, i2] > 0.0) & valid_points_xyz(pose_xyz[:, i1, :], eps) & valid_points_xyz(pose_xyz[:, i2, :], eps)
        return (pose_xyz[m, i1, :] + pose_xyz[m, i2, :]) / 2.0 if np.any(m) else None
    
    mid_hip = collect_mid(23, 24)
    if mid_hip is not None: return mid_hip.mean(axis=0)
    mid_sh = collect_mid(11, 12)
    if mid_sh is not None: return mid_sh.mean(axis=0)
    m_nose = (vis[:, 0] > 0.0) & valid_points_xyz(pose_xyz[:, 0, :], eps)
    if np.any(m_nose): return pose_xyz[m_nose, 0, :].mean(axis=0)
    return np.zeros(3, dtype=np.float32)

def compute_global_scale(pose_xyz, vis, root, eps=1e-6):
    def collect_dist(i1, i2):
        m = (vis[:, i1] > 0.0) & (vis[:, i2] > 0.0) & valid_points_xyz(pose_xyz[:, i1, :]) & valid_points_xyz(pose_xyz[:, i2, :])
        if not np.any(m): return None
        d = np.linalg.norm(pose_xyz[m, i1, :] - pose_xyz[m, i2, :], axis=1)
        return float(d[d > eps].mean()) if d[d > eps].size > 0 else None

    d_sh = collect_dist(11, 12)
    if d_sh is not None: return d_sh
    m_all = (vis > 0.0) & valid_points_xyz(pose_xyz)
    if np.any(m_all):
        d = np.linalg.norm(pose_xyz[m_all] - root[None, :], axis=1)
        if d[d > eps].size > 0: return float(d[d > eps].mean())
    return 1.0

# ============================================================================
# HAND FIXING & SMOOTHING
# ============================================================================
def frame_valid_hand(hand_t, min_pts=8, eps=1e-8): return int(np.any(np.abs(hand_t) > eps, axis=1).sum()) >= min_pts
def hand_centroid(hand_t, eps=1e-8):
    m = np.any(np.abs(hand_t) > eps, axis=1)
    return hand_t[m].mean(axis=0) if np.any(m) else None

def fix_swap_and_gate_hands(lh, rh, lw, rw, min_pts=8, hand_wrist_max_dist=1.1, eps=1e-8):
    for t in range(lh.shape[0]):
        l_ok, r_ok = frame_valid_hand(lh[t], min_pts, eps), frame_valid_hand(rh[t], min_pts, eps)
        wl_ok, wr_ok = is_valid_wrist(lw[t], eps), is_valid_wrist(rw[t], eps)
        cL, cR = hand_centroid(lh[t], eps) if l_ok else None, hand_centroid(rh[t], eps) if r_ok else None

        if l_ok and r_ok and wl_ok and wr_ok and cL is not None and cR is not None:
            if dist2(cL, rw[t]) + dist2(cR, lw[t]) + 1e-6 < dist2(cL, lw[t]) + dist2(cR, rw[t]):
                lh[t], rh[t], cL, cR = rh[t].copy(), lh[t].copy(), cR, cL

        if wl_ok and l_ok and cL is not None and dist2(cL, lw[t]) > hand_wrist_max_dist: lh[t] = 0.0
        if wr_ok and r_ok and cR is not None and dist2(cR, rw[t]) > hand_wrist_max_dist: rh[t] = 0.0

def fill_hand_gaps_wrist_relative_tiered(hand, wrist, small_gap=6, medium_gap=15, min_pts=8, rel_change_thresh=0.7, eps=1e-8):
    valid = np.array([frame_valid_hand(hand[t], min_pts=min_pts, eps=eps) for t in range(hand.shape[0])], dtype=bool)
    idx = np.where(valid)[0]
    if idx.size == 0: return

    for a, b in zip(idx[:-1], idx[1:]):
        gap = int(b - a - 1)
        if gap <= 0 or gap > medium_gap: continue
        if not (is_valid_wrist(wrist[a], eps) and is_valid_wrist(wrist[b], eps)):
            if gap <= small_gap: hand[a+1:b] = hand[a]
            continue

        rel_a, rel_b = hand[a] - wrist[a], hand[b] - wrist[b]
        if gap > small_gap:
            for t in range(a + 1, b):
                if is_valid_wrist(wrist[t], eps): hand[t] = rel_a + wrist[t]
            continue

        delta = np.linalg.norm(rel_a - rel_b, axis=1)
        rel_delta = float(np.median(delta[np.isfinite(delta)])) if delta[np.isfinite(delta)].size else 999.0

        for t in range(a + 1, b):
            if is_valid_wrist(wrist[t], eps):
                alpha = (t - a) / (b - a)
                rel = ((1.0 - alpha) * rel_a + alpha * rel_b) if rel_delta <= rel_change_thresh else rel_a
                hand[t] = rel + wrist[t]

def _alpha(cutoff_hz, dt): return float(1.0 / (1.0 + (1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-6))) / dt))
def one_euro_filter_series(x, valid, fps, min_cutoff, beta, d_cutoff):
    T, D = x.shape
    out, dt = np.zeros_like(x), 1.0 / max(fps, 1e-6)
    x_prev, x_hat_prev, dx_hat_prev, has_prev = np.zeros(D), np.zeros(D), np.zeros(D), False

    for t in range(T):
        if not valid[t]:
            has_prev = False
            continue
        xt = x[t]
        if not has_prev:
            out[t], x_prev, x_hat_prev, dx_hat_prev[:], has_prev = xt, xt, xt, 0.0, True
            continue
        dx = (xt - x_prev) / dt
        dx_hat = _alpha(d_cutoff, dt) * dx + (1.0 - _alpha(d_cutoff, dt)) * dx_hat_prev
        a = _alpha(min_cutoff + beta * np.linalg.norm(dx_hat), dt)
        x_hat = a * xt + (1.0 - a) * x_hat_prev
        out[t], x_prev, x_hat_prev, dx_hat_prev = x_hat, xt, x_hat, dx_hat
    return out

def smooth_points_over_time(pts, eps, fps, min_cutoff, beta, d_cutoff):
    for i in range(pts.shape[1]):
        x = pts[:, i, :]
        valid = valid_points_xyz(x, eps) & np.isfinite(x).all(axis=1)
        if np.any(valid): pts[:, i, :] = one_euro_filter_series(x, valid, fps, min_cutoff, beta, d_cutoff)

# ============================================================================
# MAIN PREPROCESSING & PADDING (Used by Webcam)
# ============================================================================
def preprocess_sequence_global(seq):
    y = seq.astype(np.float32, copy=True)
    pose = y[:, :config.POSE_SIZE].reshape(-1, config.POSE_LANDMARKS, config.POSE_VALS)
    face = y[:, config.POSE_SIZE:config.POSE_SIZE + config.FACE_SIZE].reshape(-1, config.FACE_LANDMARKS, config.FACE_VALS)
    lh = y[:, config.POSE_SIZE + config.FACE_SIZE:config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE].reshape(-1, config.HAND_LANDMARKS, config.HAND_VALS)
    rh = y[:, config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE:].reshape(-1, config.HAND_LANDMARKS, config.HAND_VALS)

    px, py, pz, pv = pose[:, :, 0], pose[:, :, 1], pose[:, :, 2], pose[:, :, 3]
    finite_pose = np.isfinite(pz) & np.isfinite(pv)
    pose_in_strict = in_unit_xy(px, py) & finite_pose
    pose_in_relaxed = reasonable_xy(px, py) & finite_pose
    
    critical_mask = np.zeros((pose.shape[0], config.POSE_LANDMARKS), dtype=bool)
    for i in config.CRITICAL_POSE_IDXS: critical_mask[:, i] = True
    
    pose_keep_for_transform = (pv >= 0.1) & pose_in_strict | (critical_mask & pose_in_relaxed)
    pose[~pose_keep_for_transform, :3] = 0.0
    pose[~((pv >= 0.1) & pose_in_strict), 3] = 0.0
    pose[:, config.LEG_IDXS, :] = 0.0
    
    face[~(reasonable_xy(face[:, :, 0], face[:, :, 1]) & np.isfinite(face[:, :, 2])), :3] = 0.0
    lh[~(reasonable_xy(lh[:, :, 0], lh[:, :, 1]) & np.isfinite(lh[:, :, 2])), :3] = 0.0
    rh[~(reasonable_xy(rh[:, :, 0], rh[:, :, 1]) & np.isfinite(rh[:, :, 2])), :3] = 0.0
    
    pose_xyz, vis = pose[:, :, :3], pose[:, :, 3]
    root = compute_global_root(pose_xyz, vis, eps=config.EPS)
    scale = compute_global_scale(pose_xyz, vis, root)
    
    valid_pose = pose_keep_for_transform & valid_points_xyz(pose_xyz, config.EPS)
    pose_xyz[valid_pose] = (pose_xyz[valid_pose] - root) / scale
    pose[:, :, :3] = pose_xyz
    
    for arr in (face, lh, rh):
        m = valid_points_xyz(arr, config.EPS)
        arr[m] = (arr[m] - root) / scale
    
    lw, rw = pose_xyz[:, 15, :].copy(), pose_xyz[:, 16, :].copy()
    fix_swap_and_gate_hands(lh, rh, lw, rw)
    fill_hand_gaps_wrist_relative_tiered(lh, lw)
    fill_hand_gaps_wrist_relative_tiered(rh, rw)
    
    if config.SMOOTH_POSE: smooth_points_over_time(pose[:, :, :3], config.EPS, config.FPS, config.POSE_MIN_CUTOFF, config.POSE_BETA, config.D_CUTOFF)
    if config.SMOOTH_HANDS:
        smooth_points_over_time(lh, config.EPS, config.FPS, config.HAND_MIN_CUTOFF, config.HAND_BETA, config.D_CUTOFF)
        smooth_points_over_time(rh, config.EPS, config.FPS, config.HAND_MIN_CUTOFF, config.HAND_BETA, config.D_CUTOFF)
    
    out = np.empty_like(y)
    out[:, :config.POSE_SIZE] = pose.reshape(-1, config.POSE_SIZE)
    out[:, config.POSE_SIZE:config.POSE_SIZE + config.FACE_SIZE] = face.reshape(-1, config.FACE_SIZE)
    out[:, config.POSE_SIZE + config.FACE_SIZE:config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE] = lh.reshape(-1, config.HAND_SIZE)
    out[:, config.POSE_SIZE + config.FACE_SIZE + config.HAND_SIZE:] = rh.reshape(-1, config.HAND_SIZE)
    return out

def adaptive_padding(sequence, target_frames):
    T, D = sequence.shape
    if T >= target_frames: return sequence[:target_frames], np.ones(target_frames, dtype=np.float32)
    
    padded_seq = np.zeros((target_frames, D), dtype=np.float32)
    x_orig, x_target = np.arange(T), np.linspace(0, T-1, target_frames)
    for d in range(D):
        if np.any(np.isfinite(sequence[:, d])):
            if T >= 2:
                f = interpolate.interp1d(x_orig, sequence[:, d], kind='linear', bounds_error=False, fill_value="extrapolate")
                padded_seq[:, d] = f(x_target)
            else:
                padded_seq[:, d] = sequence[0, d]
                
    mask = np.zeros(target_frames, dtype=np.float32)
    for i, target_pos in enumerate(x_target):
        mask[i] = 1.0 if 0 <= int(round(target_pos)) < T else 0.5
    return padded_seq, mask

def hybrid_frame_strategy(sequence, original_length):
    T = sequence.shape[0]
    if T < 30:
        padded_seq, mask = adaptive_padding(sequence, config.TARGET_FRAMES)
        return [padded_seq], [mask], [{'strategy': 'interpolation_padding'}]
    elif T < config.TARGET_FRAMES:
        padded_seq = np.zeros((config.TARGET_FRAMES, sequence.shape[1]), dtype=np.float32)
        padded_seq[:T] = sequence
        if T > 0: padded_seq[T:] = sequence[-1]
        mask = np.zeros(config.TARGET_FRAMES, dtype=np.float32)
        mask[:T] = 1.0
        return [padded_seq], [mask], [{'strategy': 'edge_padding'}]
    else:
        # Standard truncation for longer sequences
        return [sequence[:config.TARGET_FRAMES]], [np.ones(config.TARGET_FRAMES, dtype=np.float32)], [{'strategy': 'first_frames'}]