from __future__ import annotations

import numpy as np

from .geometry import valid_points_xyz


# ----------------------------
# One Euro Filter (optional anti-jitter)
# ----------------------------
def _alpha(cutoff_hz: float, dt: float) -> float:
    cutoff_hz = float(max(cutoff_hz, 1e-6))
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(1.0 / (1.0 + tau / dt))


def one_euro_filter_series(
    x: np.ndarray,                 # (T, D)
    valid: np.ndarray,             # (T,) bool
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    """Adaptive low-pass, resets across invalid spans."""
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
    pts: np.ndarray,       # (T,N,3)
    eps: float,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> None:
    """In-place smoothing per landmark, reset across gaps."""
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
