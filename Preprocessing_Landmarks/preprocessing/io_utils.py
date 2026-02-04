from __future__ import annotations

from pathlib import Path
import numpy as np

from .constants import FEATURE_DIM


def iter_npy_files(root: Path):
    """Yield .npy files recursively."""
    return (p for p in root.rglob("*.npy") if p.is_file())


def load_keypoints_npy(path: Path) -> np.ndarray:
    """Load a (T,438) float array from .npy; raises ValueError if shape is wrong."""
    x = np.load(path, allow_pickle=True)
    if (not isinstance(x, np.ndarray)) or x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected shape (T,{FEATURE_DIM}), got {getattr(x, 'shape', None)}")
    return x


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
