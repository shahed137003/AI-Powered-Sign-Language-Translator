import numpy as np

def compute_velocity(x: np.ndarray) -> np.ndarray:
    v = np.zeros_like(x)
    v[1:] = x[1:] - x[:-1]
    return v


def compute_acceleration(v: np.ndarray) -> np.ndarray:
    a = np.zeros_like(v)
    a[1:] = v[1:] - v[:-1]
    return a