from __future__ import annotations

import numpy as np

from .constants import POSE_SIZE, FACE_SIZE, HAND_SIZE, HAND_LANDMARKS, HAND_VALS, FEATURE_DIM
from .geometry import is_valid_wrist, dist2


def frame_valid_hand(hand_t: np.ndarray, min_pts: int = 8, eps: float = 1e-8) -> bool:
    """A frame counts as 'hand present' if it has >= min_pts non-zero landmarks."""
    nz = np.any(np.abs(hand_t) > eps, axis=1)  # (21,)
    return int(nz.sum()) >= int(min_pts)


def hand_centroid(hand_t: np.ndarray, eps: float = 1e-8):
    m = np.any(np.abs(hand_t) > eps, axis=1)
    if not np.any(m):
        return None
    return hand_t[m].mean(axis=0)
HAND_ANCHOR_IDXS = (0, 20, 8, 4)  # wrist, pinky tip, index tip, thumb tip
ANCHOR_WEIGHTS = (2.0, 1.0, 1.0, 1.0)


def _valid_xy_point(p: np.ndarray, eps: float = 1e-8) -> bool:
    return bool(np.isfinite(p[:2]).all() and np.any(np.abs(p[:2]) > eps))


def _as_pose_anchor_block(anchor_t: np.ndarray) -> np.ndarray | None:
    """
    Accept either:
      - wrist only: shape (3,)
      - multi-anchor block: shape (4,3)

    Returns a block of shape (4,3), where missing anchors are zero.
    """
    a = np.asarray(anchor_t)
    if a.ndim == 1:
        if a.shape[0] < 3:
            return None
        out = np.zeros((4, 3), dtype=a.dtype)
        out[0, :3] = a[:3]
        return out
    if a.ndim == 2:
        if a.shape[0] < 1 or a.shape[1] < 3:
            return None
        out = np.zeros((4, 3), dtype=a.dtype)
        n = min(4, a.shape[0])
        out[:n, :3] = a[:n, :3]
        return out
    return None


def _collect_anchor_pairs(
    hand_t: np.ndarray,
    pose_anchor_t: np.ndarray,
    eps: float = 1e-8,
):
    """
    Match hand anchors to same-side pose anchors:
      hand[0]  <-> pose wrist
      hand[20] <-> pose pinky
      hand[8]  <-> pose index
      hand[4]  <-> pose thumb
    """
    pose_block = _as_pose_anchor_block(pose_anchor_t)
    if pose_block is None:
        return []

    pairs = []
    for hand_idx, pose_idx, w in zip(HAND_ANCHOR_IDXS, range(4), ANCHOR_WEIGHTS):
        hp = hand_t[hand_idx]
        pp = pose_block[pose_idx]
        if _valid_xy_point(hp, eps=eps) and _valid_xy_point(pp, eps=eps):
            pairs.append((hp, pp, float(w)))
    return pairs


def _anchor_alignment_delta_xy(
    hand_t: np.ndarray,
    pose_anchor_t: np.ndarray,
    eps: float = 1e-8,
):
    """
    Returns:
      delta_xy : weighted mean XY shift to align hand anchors to pose anchors
      mean_err : weighted mean XY anchor error before/after alignment
      n_pairs  : number of valid anchor pairs used
    """
    pairs = _collect_anchor_pairs(hand_t, pose_anchor_t, eps=eps)
    if not pairs:
        return None, None, 0

    deltas = []
    errs = []
    weights = []
    for hp, pp, w in pairs:
        dxy = pp[:2] - hp[:2]
        deltas.append(dxy)
        errs.append(float(np.linalg.norm(dxy)))
        weights.append(w)

    deltas = np.asarray(deltas, dtype=np.float32)
    errs = np.asarray(errs, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    denom = float(weights.sum())
    if denom <= 1e-8:
        return None, None, 0

    delta_xy = (weights[:, None] * deltas).sum(axis=0) / denom
    mean_err = float((weights * errs).sum() / denom)
    return delta_xy, mean_err, len(pairs)


def _anchor_assignment_cost(
    hand_t: np.ndarray,
    pose_anchor_t: np.ndarray,
    eps: float = 1e-8,
) -> float | None:
    """
    Weighted mean XY anchor error for assignment scoring.
    Lower is better.
    """
    _, mean_err, n_pairs = _anchor_alignment_delta_xy(hand_t, pose_anchor_t, eps=eps)
    if n_pairs == 0:
        return None
    return mean_err

def hand_anchor0(hand_t: np.ndarray, eps: float = 1e-8):
    """Use hand landmark 0 (hand wrist) as the anchor if it is non-zero."""
    h0 = hand_t[0]
    if not np.any(np.abs(h0) > eps):
        return None
    return h0

def translate_hands_before_swap(
    lh: np.ndarray, rh: np.ndarray,
    lpose: np.ndarray, rpose: np.ndarray,
    min_pts: int = 8,
    hand_wrist_max_dist: float = 0.02,
    eps: float = 1e-8,
    clip_path: str | None = None,
) -> None:
    T = lh.shape[0]
    for t in range(T):
        _align_current_hand_to_pose_anchors_xy(
            lh, lpose, t,
            hand_name="left",
            min_pts=min_pts,
            hand_wrist_max_dist=hand_wrist_max_dist,
            eps=eps,
            clip_path=clip_path,
        )
        _align_current_hand_to_pose_anchors_xy(
            rh, rpose, t,
            hand_name="right",
            min_pts=min_pts,
            hand_wrist_max_dist=hand_wrist_max_dist,
            eps=eps,
            clip_path=clip_path,
        )

def _pose_anchor_separation_xy(
    left_pose_t: np.ndarray,
    right_pose_t: np.ndarray,
    eps: float = 1e-8,
) -> float | None:
    """
    Returns weighted mean XY separation between same-type left/right pose anchors.
    Works whether each side is:
      - wrist only: shape (3,)
      - full anchor block: shape (4,3)
    """
    lblock = _as_pose_anchor_block(left_pose_t)
    rblock = _as_pose_anchor_block(right_pose_t)
    if lblock is None or rblock is None:
        return None

    seps = []
    weights = []
    for i, w in zip(range(4), ANCHOR_WEIGHTS):
        lp = lblock[i]
        rp = rblock[i]
        if _valid_xy_point(lp, eps=eps) and _valid_xy_point(rp, eps=eps):
            seps.append(float(np.linalg.norm(lp[:2] - rp[:2])))
            weights.append(float(w))

    if not seps:
        return None

    seps = np.asarray(seps, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return float((weights * seps).sum() / max(weights.sum(), 1e-8))

def trim_edge_filler_frames(
    seq: np.ndarray,
    margin: int = 2,
    eps: float = 1e-8,
) -> np.ndarray:
    """Trim leading/trailing frames where BOTH hands are absent.

    Intended target: isolated-SLR clips with idle start/end where hands are out of frame.
    The trim is edge-only; missing hands in the middle are preserved.

    Input/Output: (T, 438) with FEATURE_DIM consistency.
    """
    if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected shape (T,{FEATURE_DIM}), got {seq.shape}")

    T = int(seq.shape[0])
    if T == 0:
        return seq
    if margin < 0:
        raise ValueError("margin must be >= 0")

    # Extract hands in raw feature space.
    lh = seq[:, POSE_SIZE + FACE_SIZE : POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(T, HAND_LANDMARKS, HAND_VALS)
    rh = seq[:, POSE_SIZE + FACE_SIZE + HAND_SIZE :].reshape(T, HAND_LANDMARKS, HAND_VALS)

    # A frame is 'active' if either hand has at least 1 non-zero landmark.
    l_present = np.array([frame_valid_hand(lh[t], min_pts=1, eps=eps) for t in range(T)], dtype=bool)
    r_present = np.array([frame_valid_hand(rh[t], min_pts=1, eps=eps) for t in range(T)], dtype=bool)
    active = l_present | r_present

    idx = np.where(active)[0]
    if idx.size == 0:
        # No usable hand frames -> keep original sequence (avoid empty output).
        return seq

    start = max(0, int(idx[0]) - int(margin))
    end = min(T - 1, int(idx[-1]) + int(margin))

    if start == 0 and end == T - 1:
        return seq
    return seq[start : end + 1]


# ----------------------------
# Sanity: swap-fix + gating
# ----------------------------
def _align_current_hand_to_pose_anchors_xy (
    hand: np.ndarray,
    pose_anchors: np.ndarray,
    t: int,
    hand_name: str = "hand",
    min_pts: int = 8,
    hand_wrist_max_dist: float = 0.5,
    eps: float = 1e-8,
    clip_path: str | None = None,
) -> None:
    """
    Apply anchor-based XY hand translation on the current frame.

    Uses same-side pose anchors (wrist, pinky, index, thumb) and matching hand anchors
    to compute a weighted XY shift. If the weighted anchor error exceeds the threshold,
    the whole visible hand is translated in XY only.
    """
    cur_ok = frame_valid_hand(hand[t], min_pts=min_pts, eps=eps)
    if not cur_ok:
        return

    delta_xy, old_d, n_pairs = _anchor_alignment_delta_xy(hand[t], pose_anchors[t], eps=eps)
    if (delta_xy is None) or (old_d is None) or (n_pairs == 0):
        return

    if old_d <= hand_wrist_max_dist:
        return

    clip_str = clip_path if clip_path is not None else "<unknown_clip>"
    m = np.any(np.abs(hand[t]) > eps, axis=1)
    if not np.any(m):
        return

    hand[t, m, 0] = hand[t, m, 0] + float(delta_xy[0])
    hand[t, m, 1] = hand[t, m, 1] + float(delta_xy[1])

    _, new_d, _ = _anchor_alignment_delta_xy(hand[t],pose_anchors[t], eps=eps)
    if new_d is None:
        new_d = -1.0

    print(
        f"[hand_repair] clip={clip_str} frame={t} hand={hand_name} "
        f"old_d={old_d:.6f} new_d={new_d:.6f} pairs={n_pairs} "
        f"dx={float(delta_xy[0]):.6f} dy={float(delta_xy[1]):.6f}"
    )

def fix_swap_and_gate_hands(
    lh: np.ndarray, rh: np.ndarray,
    left_pose_anchors: np.ndarray, right_pose_anchors: np.ndarray,
    min_pts: int = 8,
    swap_min_pose_sep: float = 0.1,
    hand_wrist_max_dist: float = 0.3,
    eps: float = 1e-8,
    clip_path: str | None = None,
) -> None:
    """
    
    Anchor-based hand swap decision.

    For each frame:
    1) require both hands to be present
    2) skip swapping if left/right pose-anchor groups are too close in XY
    3) compare same-side vs crossed pose/hand assignment costs
    4) swap only if crossed assignment is better

    """
    T = lh.shape[0]
    for t in range(T):

        l_ok = frame_valid_hand(lh[t], min_pts=min_pts, eps=eps)
        r_ok = frame_valid_hand(rh[t], min_pts=min_pts, eps=eps)

        if not (l_ok and r_ok):
            continue
        pose_sep = _pose_anchor_separation_xy(left_pose_anchors[t], right_pose_anchors[t], eps=eps)
        if (pose_sep is not None) and (pose_sep < swap_min_pose_sep):
            clip_str = clip_path if clip_path is not None else "<unknown_clip>"
            print(
                f"[hand_swap_skip_close_pose] clip={clip_str} frame={t} "
                f"pose_sep={pose_sep:.6f}"
            )
            continue
        d_ll = _anchor_assignment_cost(lh[t], left_pose_anchors[t], eps=eps)
        d_lr = _anchor_assignment_cost(lh[t], right_pose_anchors[t], eps=eps)
        d_rr = _anchor_assignment_cost(rh[t], right_pose_anchors[t], eps=eps)
        d_rl = _anchor_assignment_cost(rh[t], left_pose_anchors[t], eps=eps)

        if None in (d_ll, d_lr, d_rr, d_rl):
            continue

        if (d_lr + d_rl) + 1e-6 < (d_ll + d_rr):
            clip_str = clip_path if clip_path is not None else "<unknown_clip>"
            print(
                f"[hand_swap] clip={clip_str} frame={t} "
                f"d_ll={d_ll:.6f} d_rr={d_rr:.6f} d_lr={d_lr:.6f} d_rl={d_rl:.6f}"
            )
            lh[t], rh[t] = rh[t].copy(), lh[t].copy()
# ----------------------------
# Hand gap filling (tiered)
# ----------------------------

def _good_fill_endpoint(
    hand_t: np.ndarray,
    pose_anchor_t: np.ndarray,
    min_pts: int = 8,
    anchor_max_err: float | None = None,
    eps: float = 1e-8,
) -> bool:
    if not frame_valid_hand(hand_t, min_pts=min_pts, eps=eps):
        return False

    cost = _anchor_assignment_cost(hand_t, pose_anchor_t, eps=eps)
    if cost is None:
        return False

    if (anchor_max_err is not None) and (cost > anchor_max_err):
        return False

    return True


def _pose_anchor_delta_xy(
    src_pose_t: np.ndarray,
    dst_pose_t: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray | None:
    src_block = _as_pose_anchor_block(src_pose_t)
    dst_block = _as_pose_anchor_block(dst_pose_t)
    if src_block is None or dst_block is None:
        return None

    deltas = []
    weights = []
    for i, w in zip(range(4), ANCHOR_WEIGHTS):
        sp = src_block[i]
        dp = dst_block[i]
        if _valid_xy_point(sp, eps=eps) and _valid_xy_point(dp, eps=eps):
            deltas.append(dp[:2] - sp[:2])
            weights.append(float(w))

    if not deltas:
        return None

    deltas = np.asarray(deltas, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return (weights[:, None] * deltas).sum(axis=0) / max(float(weights.sum()), 1e-8)


def _shift_hand_xy(
    hand_t: np.ndarray,
    delta_xy: np.ndarray | None,
    eps: float = 1e-8,
) -> np.ndarray | None:
    if delta_xy is None:
        return None

    out = hand_t.copy()
    m = np.any(np.abs(out) > eps, axis=1)
    if not np.any(m):
        return None

    out[m, 0] = out[m, 0] + float(delta_xy[0])
    out[m, 1] = out[m, 1] + float(delta_xy[1])
    return out


def _blend_hands(
    hand_a: np.ndarray,
    hand_b: np.ndarray,
    alpha: float,
    eps: float = 1e-8,
) -> np.ndarray:
    out = np.zeros_like(hand_a)

    a_valid = np.any(np.abs(hand_a) > eps, axis=1)
    b_valid = np.any(np.abs(hand_b) > eps, axis=1)

    both = a_valid & b_valid
    only_a = a_valid & (~b_valid)
    only_b = b_valid & (~a_valid)

    out[both] = (1.0 - alpha) * hand_a[both] + alpha * hand_b[both]
    out[only_a] = hand_a[only_a]
    out[only_b] = hand_b[only_b]
    return out


def _anchor_relative_shape_delta(
    hand_a: np.ndarray,
    pose_a: np.ndarray,
    hand_b: np.ndarray,
    pose_b: np.ndarray,
    eps: float = 1e-8,
) -> float | None:
    pose_block_a = _as_pose_anchor_block(pose_a)
    pose_block_b = _as_pose_anchor_block(pose_b)
    if pose_block_a is None or pose_block_b is None:
        return None

    vals = []
    weights = []

    for hand_idx, pose_idx, w in zip(HAND_ANCHOR_IDXS, range(4), ANCHOR_WEIGHTS):
        ha = hand_a[hand_idx]
        hb = hand_b[hand_idx]
        pa = pose_block_a[pose_idx]
        pb = pose_block_b[pose_idx]

        if (
            _valid_xy_point(ha, eps=eps)
            and _valid_xy_point(hb, eps=eps)
            and _valid_xy_point(pa, eps=eps)
            and _valid_xy_point(pb, eps=eps)
        ):
            rel_a = ha[:2] - pa[:2]
            rel_b = hb[:2] - pb[:2]
            vals.append(float(np.linalg.norm(rel_a - rel_b)))
            weights.append(float(w))

    if not vals:
        return None

    vals = np.asarray(vals, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    return float((weights * vals).sum() / max(float(weights.sum()), 1e-8))


def fill_hand_gaps_anchor_relative_tiered(
    hand: np.ndarray,          # (T,21,3) - modified IN PLACE
    pose_anchors: np.ndarray,  # (T,4,3) preferred; wrist-only still accepted
    small_gap: int = 6,
    medium_gap: int = 15,
    min_pts: int = 8,
    rel_change_thresh: float = 0.7,
    anchor_max_err: float | None = None,
    eps: float = 1e-8,
) -> None:
    """
   
    Anchor-aware hand gap filling.

    Uses same-side pose-anchor blocks to:
    - accept only well-aligned endpoint frames for filling
    - carry hand shape forward using pose-anchor motion on medium gaps
    - interpolate short gaps only when anchor-relative hand shape is consistent
    - leave large gaps unfilled
    - perform no edge fill

    """
    T = hand.shape[0]

    valid = np.array(
        [
            _good_fill_endpoint(
                hand[t],
                pose_anchors[t],
                min_pts=min_pts,
                anchor_max_err=anchor_max_err,
                eps=eps,
            )
            for t in range(T)
        ],
        dtype=bool,
    )

    idx = np.where(valid)[0]
    if idx.size == 0:
        return

    for a, b in zip(idx[:-1], idx[1:]):
        gap = int(b - a - 1)
        if gap <= 0:
            continue
        if gap > medium_gap:
            continue

        shape_delta = _anchor_relative_shape_delta(
            hand[a], pose_anchors[a],
            hand[b], pose_anchors[b],
            eps=eps,
        )

        allow_interp = (
            gap <= small_gap
            and (shape_delta is not None)
            and (shape_delta <= rel_change_thresh)
        )

        for t in range(a + 1, b):
            delta_a = _pose_anchor_delta_xy(pose_anchors[a], pose_anchors[t], eps=eps)
            aligned_a = _shift_hand_xy(hand[a], delta_a, eps=eps)

            if allow_interp:
                delta_b = _pose_anchor_delta_xy(pose_anchors[b], pose_anchors[t], eps=eps)
                aligned_b = _shift_hand_xy(hand[b], delta_b, eps=eps)
                alpha = (t - a) / (b - a)

                if (aligned_a is not None) and (aligned_b is not None):
                    hand[t] = _blend_hands(aligned_a, aligned_b, alpha, eps=eps)
                elif aligned_a is not None:
                    hand[t] = aligned_a
                elif aligned_b is not None:
                    hand[t] = aligned_b
            else:
                if aligned_a is not None:
                    hand[t] = aligned_a