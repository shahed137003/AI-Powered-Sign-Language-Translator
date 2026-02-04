from __future__ import annotations

import numpy as np

from .geometry import valid_points_xyz


# ----------------------------
# Global root/scale (mean over frames)
# ----------------------------
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
