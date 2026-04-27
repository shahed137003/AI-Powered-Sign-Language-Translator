import numpy as np

def compute_angle(a, b, c, eps=1e-6):
    ba = a - b
    bc = c - b

    dot = np.sum(ba * bc, axis=-1)
    norm = (np.linalg.norm(ba, axis=-1) *
            np.linalg.norm(bc, axis=-1)) + eps

    cos = np.clip(dot / norm, -1.0, 1.0)
    return np.arccos(cos)


def compute_hand_angles(hand: np.ndarray):
    chains = [
        (0,1,2),(1,2,3),(2,3,4),
        (0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20),
    ]

    angles = [compute_angle(hand[:,a], hand[:,b], hand[:,c])[:,None]
              for a,b,c in chains]

    return np.concatenate(angles, axis=-1)