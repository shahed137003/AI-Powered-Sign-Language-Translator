import numpy as np

class Config:
    TARGET_FRAMES = 158
    FEATURE_DIM = 438
    # Exact 60 indices for Lips and Eyebrows used in training
    FACE_IDXS = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
        107, 66, 105, 63, 70, 336, 296, 334, 293, 300, 168, 193, 189, 221, 222, 223, 224, 225, 113, 226
    ]

def preprocess_sequence_global(seq):
    y = seq.astype(np.float32, copy=True)
    T = y.shape[0]
    
    pose = y[:, :132].reshape(-1, 33, 4)
    face = y[:, 132:312].reshape(-1, 60, 3)
    lh = y[:, 312:375].reshape(-1, 21, 3)
    rh = y[:, 375:].reshape(-1, 21, 3)

    for t in range(T):
        # 1. ROOT CENTERING (Shoulder midpoint)
        if pose[t, 11, 3] > 0.1 and pose[t, 12, 3] > 0.1:
            root = (pose[t, 11, :3] + pose[t, 12, :3]) / 2.0
            
            # 2. SHOULDER SCALING (Section 6.3)
            # Calculate distance between shoulders to normalize size
            sh_dist = np.linalg.norm(pose[t, 11, :3] - pose[t, 12, :3])
            scale = sh_dist if sh_dist > 1e-6 else 1.0
        else:
            root = pose[t, 0, :3]
            scale = 1.0
            
        # Apply centering and scaling
        pose[t, :, :3] = (pose[t, :, :3] - root) / scale
        face[t] = (face[t] - root) / scale
        lh[t] = (lh[t] - root) / scale
        rh[t] = (rh[t] - root) / scale

        # 3. VISIBILITY GATING
        pose[t, pose[t, :, 3] < 0.5, :3] = 0.0

    return np.concatenate([
        pose.reshape(T, -1), face.reshape(T, -1),
        lh.reshape(T, -1), rh.reshape(T, -1)
    ], axis=1)