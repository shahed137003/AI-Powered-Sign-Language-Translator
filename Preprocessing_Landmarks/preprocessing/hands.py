from __future__ import annotations

import numpy as np

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


# ----------------------------
# Sanity: swap-fix + gating
# ----------------------------
def fix_swap_and_gate_hands(
    lh: np.ndarray, rh: np.ndarray,   # (T,21,3) modified IN PLACE
    lw: np.ndarray, rw: np.ndarray,   # (T,3)
    min_pts: int = 8,
    hand_wrist_max_dist: float = 1.1,
    eps: float = 1e-8,
) -> None:
    """
    1) If both hands present and look swapped, swap them for that frame.
    2) If a hand centroid is too far from its wrist, zero it.
    """
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


# ----------------------------
# Hand gap filling (tiered)
# ----------------------------
def fill_hand_gaps_wrist_relative_tiered(
    hand: np.ndarray,          # (T,21,3) - modified IN PLACE
    wrist: np.ndarray,         # (T,3)
    small_gap: int = 6,        # your old max_gap behavior
    medium_gap: int = 15,      # carry-forward wrist-relative in (small_gap, medium_gap]
    min_pts: int = 8,
    rel_change_thresh: float = 0.7,
    eps: float = 1e-8,
) -> None:
    """
    Small gaps (<= small_gap):
      - If endpoints consistent: wrist-relative interpolation
      - Else: wrist-relative carry-forward

    Medium gaps (small_gap < gap <= medium_gap):
      - wrist-relative carry-forward ONLY (semi-static hand follows wrist)

    Large gaps (> medium_gap):
      - do nothing (avoid hallucinating long missing spans)

    No edge fill.
    """
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
            continue  # large gap -> leave missing

        if not (is_valid_wrist(wrist[a], eps=eps) and is_valid_wrist(wrist[b], eps=eps)):
            # cannot do wrist-relative safely -> keep old absolute copy for short gaps only
            if gap <= small_gap:
                for t in range(a + 1, b):
                    hand[t] = hand[a]
            continue

        rel_a = hand[a] - wrist[a]
        rel_b = hand[b] - wrist[b]

        # medium gap: always carry-forward rel_a (no finger interpolation)
        if gap > small_gap:
            for t in range(a + 1, b):
                set_from_rel(t, rel_a)
            continue

        # small gap: decide interpolate vs carry based on shape change
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
