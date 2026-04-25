import numpy as np

def temporal_encoding(T: int):
    t = np.arange(T)
    t_norm = t / max(T - 1, 1)

    return np.stack([
        t_norm,
        np.sin(2 * np.pi * t_norm),
        np.cos(2 * np.pi * t_norm)
    ], axis=-1)