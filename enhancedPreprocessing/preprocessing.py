# ============================================================
# hand_swap_preprocessor.py
# Enhanced Hand Swap Detection Preprocessor Class
# ============================================================

import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================

POSE_LANDMARKS, POSE_VALS = 33, 4
HAND_LANDMARKS, HAND_VALS = 21, 3
FACE_LANDMARKS, FACE_VALS = 60, 3

POSE_SIZE = POSE_LANDMARKS * POSE_VALS          # 132
HAND_SIZE = HAND_LANDMARKS * HAND_VALS          # 63
FACE_SIZE = FACE_LANDMARKS * FACE_VALS          # 180
FEATURE_DIM = POSE_SIZE + FACE_SIZE + 2 * HAND_SIZE  # 438

# Legs to drop (knees->feet). Hips (23,24) are kept.
LEG_IDXS = list(range(25, 33))

# Critical joints we keep for transform even if visibility is low
CRITICAL_POSE_IDXS = {0, 11, 12, 13, 14, 15, 16, 23, 24}


class HandSwapPreprocessor:
    """
    Enhanced hand swap detection and preprocessing for sign language landmarks
    
    This class handles:
    - Hand swap detection with false positive reduction
    - Hand gap filling
    - Pose normalization
    - Padding and mask generation
    
    Usage:
        preprocessor = HandSwapPreprocessor()
        processed_data, masks, labels, swapped_info = preprocessor.process_dataset(
            cleaned_data, cleaned_labels
        )
    """
    
    def __init__(self, 
                 use_3d: bool = True,
                 confidence_threshold: float = 0.55,
                 temporal_window: int = 7,
                 swap_duration_threshold: int = 3,
                 iou_threshold: float = 0.25,
                 min_hand_pts: int = 5,
                 hand_wrist_max_dist: float = 1.1,
                 small_gap: int = 6,
                 medium_gap: int = 15,
                 rel_change_thresh: float = 0.7,
                 pose_vis_thresh: float = 0.1,
                 keep_legs: bool = False,
                 smooth: bool = False,
                 verbose: bool = True):
        """
        Initialize the hand swap preprocessor
        
        Args:
            use_3d: Use 3D distances for better accuracy
            confidence_threshold: Minimum confidence to apply swap (higher = fewer false positives)
            temporal_window: Number of frames for temporal smoothing
            swap_duration_threshold: Minimum consecutive frames to confirm swap
            iou_threshold: IoU threshold for hand overlap detection
            min_hand_pts: Minimum landmarks for hand to be considered valid
            hand_wrist_max_dist: Max distance for hand-wrist association
            small_gap: Maximum gap for interpolation
            medium_gap: Maximum gap for carry-forward
            rel_change_thresh: Threshold for shape change detection
            pose_vis_thresh: Pose visibility threshold
            keep_legs: Whether to keep leg landmarks
            smooth: Apply smoothing
            verbose: Print progress information
        """
        self.use_3d = use_3d
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.swap_duration_threshold = swap_duration_threshold
        self.iou_threshold = iou_threshold
        self.min_hand_pts = min_hand_pts
        self.hand_wrist_max_dist = hand_wrist_max_dist
        self.small_gap = small_gap
        self.medium_gap = medium_gap
        self.rel_change_thresh = rel_change_thresh
        self.pose_vis_thresh = pose_vis_thresh
        self.keep_legs = keep_legs
        self.smooth = smooth
        self.verbose = verbose
        
        # Smoothing parameters
        self.smooth_fps = 20.0
        self.smooth_pose = True
        self.smooth_hands = True
        self.smooth_face = False
        self.pose_min_cutoff = 1.5
        self.pose_beta = 0.6
        self.hand_min_cutoff = 3.0
        self.hand_beta = 0.8
        self.face_min_cutoff = 2.0
        self.face_beta = 0.6
        self.d_cutoff = 1.0
        self.eps = 1e-8
        
        # Stats tracking
        self.videos_with_swaps = []
        self.swap_counts = []
        
    # ============================================================
    # Geometry Functions
    # ============================================================
    
    def _in_unit_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (np.isfinite(x) & np.isfinite(y) & 
                (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0))
    
    def _reasonable_xy(self, x: np.ndarray, y: np.ndarray, lo: float = -0.25, hi: float = 1.25) -> np.ndarray:
        return (np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi))
    
    def _valid_points_xyz(self, arr: np.ndarray) -> np.ndarray:
        return np.any(np.abs(arr) > self.eps, axis=-1)
    
    def _is_valid_wrist(self, w: np.ndarray) -> bool:
        return bool(np.isfinite(w).all() and np.any(np.abs(w) > self.eps))
    
    def _dist2(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))
    
    def _dist3(self, a: np.ndarray, b: np.ndarray) -> float:
        """3D distance if available"""
        if self.use_3d and len(a) >= 3 and len(b) >= 3:
            return float(np.linalg.norm(a[:3] - b[:3]))
        return float(np.linalg.norm(a[:2] - b[:2]))
    
    # ============================================================
    # Hand Detection Functions
    # ============================================================
    
    def _frame_valid_hand(self, hand_t: np.ndarray, min_pts: int = None) -> bool:
        """Check if hand has enough valid landmarks"""
        if min_pts is None:
            min_pts = self.min_hand_pts
        nz = np.any(np.abs(hand_t) > self.eps, axis=1)
        return int(nz.sum()) >= int(min_pts)
    
    def _hand_centroid(self, hand_t: np.ndarray):
        """Compute hand centroid"""
        m = np.any(np.abs(hand_t) > self.eps, axis=1)
        if not np.any(m):
            return None
        return hand_t[m].mean(axis=0)
    
    def _compute_hand_overlap(self, lh, rh) -> float:
        """Compute IoU between hand bounding boxes"""
        def get_bbox(hand):
            valid_pts = hand[np.any(np.abs(hand) > self.eps, axis=1)]
            if len(valid_pts) == 0:
                return None
            min_x, min_y = valid_pts[:, 0].min(), valid_pts[:, 1].min()
            max_x, max_y = valid_pts[:, 0].max(), valid_pts[:, 1].max()
            return (min_x, min_y, max_x, max_y)
        
        l_bbox = get_bbox(lh)
        r_bbox = get_bbox(rh)
        
        if l_bbox is None or r_bbox is None:
            return 0.0
        
        lx1, ly1, lx2, ly2 = l_bbox
        rx1, ry1, rx2, ry2 = r_bbox
        
        ix1 = max(lx1, rx1)
        iy1 = max(ly1, ry1)
        ix2 = min(lx2, rx2)
        iy2 = min(ly2, ry2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_l = (lx2 - lx1) * (ly2 - ly1)
        area_r = (rx2 - rx1) * (ry2 - ry1)
        union = area_l + area_r - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_hand_symmetry(self, lh, rh) -> float:
        """Check if hands are symmetrically positioned"""
        l_center = self._hand_centroid(lh)
        r_center = self._hand_centroid(rh)
        
        if l_center is None or r_center is None:
            return 0.0
        
        body_center = 0.5
        l_dist = abs(l_center[0] - body_center)
        r_dist = abs(r_center[0] - body_center)
        
        symmetry = 1.0 - abs(l_dist - r_dist) / (l_dist + r_dist + self.eps)
        return symmetry
    
    # ============================================================
    # Improved Hand Swap Detector Inner Class
    # ============================================================
    
    class _SwapDetector:
        def __init__(self, parent):
            self.parent = parent
            self.swap_history = deque(maxlen=parent.temporal_window)
            self.consecutive_swaps = 0
            
        def compute_swap_confidence(self, lh, rh, lw, rw):
            l_valid = self.parent._frame_valid_hand(lh)
            r_valid = self.parent._frame_valid_hand(rh)
            
            if not (l_valid and r_valid):
                return False, 0.0, 0.0
            
            overlap = self.parent._compute_hand_overlap(lh, rh)
            if overlap > self.parent.iou_threshold:
                return False, 0.0, overlap
            
            cL = self.parent._hand_centroid(lh)
            cR = self.parent._hand_centroid(rh)
            
            if cL is None or cR is None:
                return False, 0.0, 0.0
            
            # Centroid-based detection
            d_ll = self.parent._dist3(cL, lw)
            d_lr = self.parent._dist3(cL, rw)
            d_rr = self.parent._dist3(cR, rw)
            d_rl = self.parent._dist3(cR, lw)
            
            original_dist = d_ll + d_rr
            swapped_dist = d_lr + d_rl
            
            centroid_swap = swapped_dist < (original_dist * 0.85)
            centroid_confidence = max(0.0, min(1.0, (original_dist - swapped_dist) / (original_dist + self.parent.eps)))
            
            # Keypoint-based detection
            lh_points = lh[np.any(np.abs(lh) > self.parent.eps, axis=1)]
            rh_points = rh[np.any(np.abs(rh) > self.parent.eps, axis=1)]
            
            if len(lh_points) > 3 and len(rh_points) > 3:
                lh_to_lw = np.median([self.parent._dist3(p, lw) for p in lh_points])
                lh_to_rw = np.median([self.parent._dist3(p, rw) for p in lh_points])
                rh_to_lw = np.median([self.parent._dist3(p, lw) for p in rh_points])
                rh_to_rw = np.median([self.parent._dist3(p, rw) for p in rh_points])
                
                original_key_dist = lh_to_lw + rh_to_rw
                swapped_key_dist = lh_to_rw + rh_to_lw
                
                keypoint_swap = swapped_key_dist < (original_key_dist * 0.85)
                keypoint_confidence = max(0.0, min(1.0, (original_key_dist - swapped_key_dist) / (original_key_dist + self.parent.eps)))
            else:
                keypoint_swap = centroid_swap
                keypoint_confidence = centroid_confidence
            
            symmetry = self.parent._compute_hand_symmetry(lh, rh)
            symmetry_penalty = 1.0 - (symmetry * 0.5)
            
            confidence = (centroid_confidence * 0.3 + keypoint_confidence * 0.5) * symmetry_penalty
            should_swap = centroid_swap and keypoint_swap and confidence > self.parent.confidence_threshold
            
            return should_swap, confidence, overlap
        
        def detect_and_fix_swap(self, lh, rh, lw, rw, frame_idx):
            should_swap, confidence, overlap = self.compute_swap_confidence(lh, rh, lw, rw)
            
            self.swap_history.append(should_swap)
            
            if len(self.swap_history) >= self.parent.temporal_window // 2:
                recent_swaps = sum(list(self.swap_history)[-self.parent.temporal_window:])
                if recent_swaps > self.parent.temporal_window // 2:
                    self.consecutive_swaps += 1
                else:
                    self.consecutive_swaps = 0
                
                temporal_swap = self.consecutive_swaps >= self.parent.swap_duration_threshold
            else:
                temporal_swap = False
            
            if temporal_swap and confidence > self.parent.confidence_threshold:
                return rh.copy(), lh.copy(), True
            
            return lh, rh, False
    
    # ============================================================
    # Normalization Functions
    # ============================================================
    
    def _compute_global_root(self, pose_xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
        def collect_mid(i1: int, i2: int):
            m = (vis[:, i1] > 0.0) & (vis[:, i2] > 0.0)
            m = m & self._valid_points_xyz(pose_xyz[:, i1, :]) & self._valid_points_xyz(pose_xyz[:, i2, :])
            if not np.any(m):
                return None
            return (pose_xyz[m, i1, :] + pose_xyz[m, i2, :]) / 2.0

        mid_hip = collect_mid(23, 24)
        if mid_hip is not None:
            return mid_hip.mean(axis=0)
        mid_sh = collect_mid(11, 12)
        if mid_sh is not None:
            return mid_sh.mean(axis=0)
        m_nose = (vis[:, 0] > 0.0) & self._valid_points_xyz(pose_xyz[:, 0, :])
        if np.any(m_nose):
            return pose_xyz[m_nose, 0, :].mean(axis=0)
        m_all = (vis > 0.0) & self._valid_points_xyz(pose_xyz)
        if np.any(m_all):
            return pose_xyz[m_all].mean(axis=0)
        return np.zeros(3, dtype=np.float32)
    
    def _compute_global_scale(self, pose_xyz: np.ndarray, vis: np.ndarray, root: np.ndarray) -> float:
        def collect_dist(i1: int, i2: int):
            m = (vis[:, i1] > 0.0) & (vis[:, i2] > 0.0)
            m = m & self._valid_points_xyz(pose_xyz[:, i1, :]) & self._valid_points_xyz(pose_xyz[:, i2, :])
            if not np.any(m):
                return None
            d = np.linalg.norm(pose_xyz[m, i1, :] - pose_xyz[m, i2, :], axis=1)
            d = d[d > 1e-6]
            return d if d.size > 0 else None

        d_sh = collect_dist(11, 12)
        if d_sh is not None:
            return float(d_sh.mean())
        d_hip = collect_dist(23, 24)
        if d_hip is not None:
            return float(d_hip.mean())
        m_all = (vis > 0.0) & self._valid_points_xyz(pose_xyz)
        if np.any(m_all):
            d = np.linalg.norm(pose_xyz[m_all] - root[None, :], axis=1)
            d = d[d > 1e-6]
            if d.size > 0:
                return float(d.mean())
        return 1.0
    
    # ============================================================
    # Smoothing Functions
    # ============================================================
    
    def _alpha(self, cutoff_hz: float, dt: float) -> float:
        cutoff_hz = float(max(cutoff_hz, 1e-6))
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return float(1.0 / (1.0 + tau / dt))
    
    def _one_euro_filter_series(self, x: np.ndarray, valid: np.ndarray, fps: float,
                                 min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
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
            a_d = self._alpha(d_cutoff, dt)
            dx_hat = a_d * dx + (1.0 - a_d) * dx_hat_prev
            cutoff = float(min_cutoff + beta * np.linalg.norm(dx_hat))
            a = self._alpha(cutoff, dt)
            x_hat = a * xt + (1.0 - a) * x_hat_prev
            out[t] = x_hat
            x_prev = xt
            x_hat_prev = x_hat
            dx_hat_prev = dx_hat
        return out
    
    def _smooth_points_over_time(self, pts: np.ndarray, fps: float,
                                  min_cutoff: float, beta: float, d_cutoff: float) -> None:
        T, N, _ = pts.shape
        for i in range(N):
            x = pts[:, i, :]
            valid = self._valid_points_xyz(x) & np.isfinite(x).all(axis=1)
            if not np.any(valid):
                continue
            pts[:, i, :] = self._one_euro_filter_series(x, valid=valid, fps=fps,
                                                        min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    
    # ============================================================
    # Hand Gap Filling
    # ============================================================
    
    def _fill_hand_gaps(self, hand: np.ndarray, wrist: np.ndarray) -> None:
        T = hand.shape[0]
        valid = np.array([self._frame_valid_hand(hand[t]) for t in range(T)], dtype=bool)
        idx = np.where(valid)[0]
        if idx.size == 0:
            return

        def set_from_rel(t: int, rel: np.ndarray, confidence: float = 1.0):
            if self._is_valid_wrist(wrist[t]):
                if confidence < 0.5:
                    hand[t] = rel * 0.7 + wrist[t]
                else:
                    hand[t] = rel + wrist[t]

        for a, b in zip(idx[:-1], idx[1:]):
            gap = int(b - a - 1)
            if gap <= 0 or gap > self.medium_gap:
                continue

            if not (self._is_valid_wrist(wrist[a]) and self._is_valid_wrist(wrist[b])):
                if gap <= self.small_gap:
                    for t in range(a + 1, b):
                        hand[t] = hand[a]
                continue

            rel_a = hand[a] - wrist[a]
            rel_b = hand[b] - wrist[b]
            confidence = 1.0 - (gap / self.medium_gap)

            if gap > self.small_gap:
                for t in range(a + 1, b):
                    set_from_rel(t, rel_a, confidence)
                continue

            delta = np.linalg.norm(rel_a - rel_b, axis=1)
            delta = delta[np.isfinite(delta)]
            rel_delta = float(np.median(delta)) if delta.size else 999.0

            if rel_delta <= self.rel_change_thresh:
                for t in range(a + 1, b):
                    alpha = (t - a) / (b - a)
                    rel = (1.0 - alpha) * rel_a + alpha * rel_b
                    set_from_rel(t, rel, confidence)
            else:
                for t in range(a + 1, b):
                    set_from_rel(t, rel_a, confidence)
    
    # ============================================================
    # Main Preprocessing Function
    # ============================================================
    
    def _preprocess_sequence(self, seq: np.ndarray) -> Tuple[np.ndarray, int]:
        """Preprocess a single sequence and return swap count"""
        y = seq.astype(np.float32, copy=True)
        if y.ndim != 2 or y.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected shape (T,{FEATURE_DIM}), got {y.shape}")

        # Trim edge filler frames
        T = int(y.shape[0])
        if T > 0:
            lh = y[:, POSE_SIZE + FACE_SIZE:POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(T, HAND_LANDMARKS, HAND_VALS)
            rh = y[:, POSE_SIZE + FACE_SIZE + HAND_SIZE:].reshape(T, HAND_LANDMARKS, HAND_VALS)
            l_present = np.array([self._frame_valid_hand(lh[t], min_pts=1) for t in range(T)], dtype=bool)
            r_present = np.array([self._frame_valid_hand(rh[t], min_pts=1) for t in range(T)], dtype=bool)
            active = l_present | r_present
            idx = np.where(active)[0]
            if idx.size > 0:
                start = max(0, int(idx[0]) - 2)
                end = min(T - 1, int(idx[-1]) + 2)
                if start != 0 or end != T - 1:
                    y = y[start:end + 1]

        # Split into components
        pose = y[:, :POSE_SIZE].reshape(-1, POSE_LANDMARKS, POSE_VALS)
        face = y[:, POSE_SIZE:POSE_SIZE + FACE_SIZE].reshape(-1, FACE_LANDMARKS, FACE_VALS)
        lh = y[:, POSE_SIZE + FACE_SIZE:POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(-1, HAND_LANDMARKS, HAND_VALS)
        rh = y[:, POSE_SIZE + FACE_SIZE + HAND_SIZE:].reshape(-1, HAND_LANDMARKS, HAND_VALS)

        # Pose cleaning
        px, py, pz, pv = pose[:, :, 0], pose[:, :, 1], pose[:, :, 2], pose[:, :, 3]
        finite_pose = np.isfinite(pz) & np.isfinite(pv)
        pose_in_strict = self._in_unit_xy(px, py) & finite_pose
        pose_in_relaxed = self._reasonable_xy(px, py) & finite_pose
        critical_mask = np.zeros((pose.shape[0], POSE_LANDMARKS), dtype=bool)
        for i in CRITICAL_POSE_IDXS:
            critical_mask[:, i] = True
        pose_keep_for_transform = (pv >= self.pose_vis_thresh) & pose_in_strict
        pose_keep_for_transform = pose_keep_for_transform | (critical_mask & pose_in_relaxed)
        pose_keep_visible = (pv >= self.pose_vis_thresh) & pose_in_strict
        bad_xyz = ~pose_keep_for_transform
        pose[bad_xyz, :3] = 0.0
        pose[~pose_keep_visible, 3] = 0.0
        if not self.keep_legs:
            pose[:, LEG_IDXS, :3] = 0.0
            pose[:, LEG_IDXS, 3] = 0.0

        # Face cleaning
        fx, fy, fz = face[:, :, 0], face[:, :, 1], face[:, :, 2]
        face_in = self._reasonable_xy(fx, fy) & np.isfinite(fz)
        face[~face_in, :3] = 0.0

        # Hands cleaning
        lx, ly, lz = lh[:, :, 0], lh[:, :, 1], lh[:, :, 2]
        lh_in = self._reasonable_xy(lx, ly) & np.isfinite(lz)
        lh[~lh_in, :3] = 0.0
        rx, ry, rz = rh[:, :, 0], rh[:, :, 1], rh[:, :, 2]
        rh_in = self._reasonable_xy(rx, ry) & np.isfinite(rz)
        rh[~rh_in, :3] = 0.0

        # Global normalization
        pose_xyz = pose[:, :, :3]
        vis = pose[:, :, 3]
        root = self._compute_global_root(pose_xyz, vis)
        scale = self._compute_global_scale(pose_xyz, vis, root)
        pose_valid_for_transform = pose_keep_for_transform & self._valid_points_xyz(pose_xyz)
        pose_xyz[pose_valid_for_transform] = (pose_xyz[pose_valid_for_transform] - root) / scale
        pose[:, :, :3] = pose_xyz
        for arr in (face, lh, rh):
            m = self._valid_points_xyz(arr)
            arr[m] = (arr[m] - root) / scale

        # Wrists
        lw = pose_xyz[:, 15, :].copy()
        rw = pose_xyz[:, 16, :].copy()

        # Hand swap detection
        swap_count = 0
        detector = self._SwapDetector(self)
        
        for t in range(len(lh)):
            l_ok = self._frame_valid_hand(lh[t])
            r_ok = self._frame_valid_hand(rh[t])
            wl_ok = self._is_valid_wrist(lw[t])
            wr_ok = self._is_valid_wrist(rw[t])
            
            cL = self._hand_centroid(lh[t]) if l_ok else None
            cR = self._hand_centroid(rh[t]) if r_ok else None
            
            if l_ok and r_ok and wl_ok and wr_ok:
                new_lh, new_rh, swapped = detector.detect_and_fix_swap(lh[t], rh[t], lw[t], rw[t], t)
                if swapped:
                    lh[t], rh[t] = new_lh, new_rh
                    swap_count += 1
                    cL = self._hand_centroid(lh[t])
                    cR = self._hand_centroid(rh[t])
            
            if wl_ok and l_ok and (cL is not None):
                if self._dist2(cL, lw[t]) > self.hand_wrist_max_dist:
                    lh[t] = 0.0
            if wr_ok and r_ok and (cR is not None):
                if self._dist2(cR, rw[t]) > self.hand_wrist_max_dist:
                    rh[t] = 0.0

        # Hand gap filling
        self._fill_hand_gaps(lh, lw)
        self._fill_hand_gaps(rh, rw)

        # Smoothing
        if self.smooth:
            if self.smooth_pose:
                self._smooth_points_over_time(pose[:, :, :3], self.smooth_fps,
                                              self.pose_min_cutoff, self.pose_beta, self.d_cutoff)
            if self.smooth_hands:
                self._smooth_points_over_time(lh, self.smooth_fps,
                                              self.hand_min_cutoff, self.hand_beta, self.d_cutoff)
                self._smooth_points_over_time(rh, self.smooth_fps,
                                              self.hand_min_cutoff, self.hand_beta, self.d_cutoff)
            if self.smooth_face:
                self._smooth_points_over_time(face, self.smooth_fps,
                                              self.face_min_cutoff, self.face_beta, self.d_cutoff)

        # Reconstruct
        out = np.empty_like(y, dtype=np.float32)
        out[:, :POSE_SIZE] = pose.reshape(-1, POSE_SIZE)
        out[:, POSE_SIZE:POSE_SIZE + FACE_SIZE] = face.reshape(-1, FACE_SIZE)
        out[:, POSE_SIZE + FACE_SIZE:POSE_SIZE + FACE_SIZE + HAND_SIZE] = lh.reshape(-1, HAND_SIZE)
        out[:, POSE_SIZE + FACE_SIZE + HAND_SIZE:] = rh.reshape(-1, HAND_SIZE)
        
        return out, swap_count
    
    # ============================================================
    # Padding and Mask Functions
    # ============================================================
    
    def _pad_to_max_frames(self, landmarks: np.ndarray, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad landmarks to max_frames and create mask"""
        current_frames = landmarks.shape[0]
        if current_frames >= max_frames:
            return landmarks[:max_frames], np.ones(max_frames, dtype=np.float32)
        
        pad_length = max_frames - current_frames
        pad = np.zeros((pad_length, FEATURE_DIM), dtype=landmarks.dtype)
        padded_landmarks = np.vstack([landmarks, pad])
        mask = np.zeros(max_frames, dtype=np.float32)
        mask[:current_frames] = 1.0
        return padded_landmarks, mask
    
    # ============================================================
    # Public Methods
    # ============================================================
    
    def process_dataset(self, 
                        cleaned_data: np.ndarray,
                        cleaned_labels: np.ndarray,
                        max_frames: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Process entire dataset with enhanced hand swap detection
        
        Args:
            cleaned_data: Input sequences array
            cleaned_labels: Corresponding labels
            max_frames: Target number of frames (auto-detect if None)
        
        Returns:
            Tuple of (processed_data, processed_masks, processed_labels, swapped_videos, swap_counts)
        """
        # Reset stats
        self.videos_with_swaps = []
        self.swap_counts = []
        
        # Calculate max frames if not provided
        if max_frames is None:
            frame_lengths = [seq.shape[0] for seq in cleaned_data]
            max_frames = max(frame_lengths)
            if self.verbose:
                print(f" Auto-detected max_frames: {max_frames}")
        
        if self.verbose:
            print(f"\n Processing {len(cleaned_data)} sequences...")
            print(f"   Target frames: {max_frames}")
            print(f"   Confidence threshold: {self.confidence_threshold}")
            print(f"   Temporal window: {self.temporal_window}")
            print("="*60)
        
        processed_data = []
        processed_masks = []
        processed_labels = []
        
        for i, (seq, label) in enumerate(tqdm(zip(cleaned_data, cleaned_labels), 
                                              total=len(cleaned_data), 
                                              disable=not self.verbose)):
            try:
                # Preprocess sequence
                seq, swap_count = self._preprocess_sequence(seq)
                
                # Pad to max frames and create mask
                padded_seq, mask = self._pad_to_max_frames(seq, max_frames)
                
                processed_data.append(padded_seq)
                processed_masks.append(mask)
                processed_labels.append(label)
                
                if swap_count > 0:
                    self.videos_with_swaps.append(i)
                    self.swap_counts.append(swap_count)
                
            except Exception as e:
                print(f"⚠️ Error processing sequence {i}: {e}")
                continue
        
        # Convert to numpy arrays
        processed_data = np.array(processed_data, dtype=np.float32)
        processed_masks = np.array(processed_masks, dtype=np.float32)
        processed_labels = np.array(processed_labels)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        
        if self.videos_with_swaps:
            print(f"\n HAND SWAP DETECTED in {len(self.videos_with_swaps)} videos:")
            print("-"*40)
            for idx, swap_count in zip(self.videos_with_swaps, self.swap_counts):
                label = cleaned_labels[idx]
                print(f"   Video #{idx}: '{label}' - {swap_count} frames swapped")
            print("-"*40)
        else:
            print(f"\n✅ NO hand swaps detected in any video!")
        
        print(f"\n Final Statistics:")
        print(f"   Total videos processed: {len(processed_data)}")
        print(f"   Videos with swaps: {len(self.videos_with_swaps)}")
        print(f"   Processed data shape: {processed_data.shape}")
        print(f"   Processed masks shape: {processed_masks.shape}")
        print(f"   Valid frames proportion: {processed_masks.mean():.3f}")
        
        return processed_data, processed_masks, processed_labels, self.videos_with_swaps, self.swap_counts
    
    def get_swapped_videos(self) -> Tuple[List[int], List[int]]:
        """Get indices and counts of videos that had hand swaps"""
        return self.videos_with_swaps, self.swap_counts
    
    def print_swap_summary(self):
        """Print summary of hand swap detections"""
        if self.videos_with_swaps:
            print(f"\n Hand Swap Summary:")
            for idx, count in zip(self.videos_with_swaps, self.swap_counts):
                print(f"   Video {idx}: {count} frames swapped")
        else:
            print("\nNo hand swaps detected")