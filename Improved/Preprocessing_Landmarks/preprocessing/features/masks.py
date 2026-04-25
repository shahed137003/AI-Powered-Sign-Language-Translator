import numpy as np

def compute_mask(x: np.ndarray, eps=1e-6):
    return (np.abs(x) > eps).astype(np.float32)


def compute_frame_validity(x: np.ndarray):
    valid = np.sum(np.abs(x) > 0, axis=1, keepdims=True)
    return valid / x.shape[1]