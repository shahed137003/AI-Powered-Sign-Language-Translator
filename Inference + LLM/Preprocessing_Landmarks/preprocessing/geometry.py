from __future__ import annotations

import numpy as np


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
