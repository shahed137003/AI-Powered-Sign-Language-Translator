
import numpy as np
### hand reconstrucation 

def initialize_first_last_flat(hand_frames):
    """
    hand_frames: (T, F) where F = 63 flattened features
    Missing hand = all zeros
    """

    T = len(hand_frames)

    # find indices with valid (non-zero) hand detections
    detected = [i for i in range(T) if not np.all(hand_frames[i] == 0)]

    if len(detected) == 0:
        return hand_frames  # nothing to reconstruct

    # average detected hand frames
    avg_hand = np.mean([hand_frames[i] for i in detected], axis=0)

    # first frame
    if np.all(hand_frames[0] == 0):
        hand_frames[0] = avg_hand

    # last frame
    if np.all(hand_frames[-1] == 0):
        hand_frames[-1] = avg_hand

    return hand_frames


def reconstruct_hands_flat(hand_frames):
    """
    Bilinear interpolation for flattened hand features.
    hand_frames: (T, F) where F = 63
    """

    hand_frames = initialize_first_last_flat(hand_frames)
    T = len(hand_frames)

    for k in range(T):
        if not np.all(hand_frames[k] == 0):
            continue 

        
        a = 1
        while k - a >= 0 and np.all(hand_frames[k - a] == 0):
            a += 1

        # find next non-zero frame
        b = 1
        while k + b < T and np.all(hand_frames[k + b] == 0):
            b += 1

        
        if k - a < 0 or k + b >= T:
            continue

        prev_frame = hand_frames[k - a]
        next_frame = hand_frames[k + b]

        
        hand_frames[k] = (b * prev_frame + a * next_frame) / (a + b)

    return hand_frames


## Data augmentation 

## optional may destroy the data 


# --------------------------------------------------
# 1) Small random noise (jitter)
# --------------------------------------------------
def augment_jitter(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

# --------------------------------------------------
# 2) Random scaling
# --------------------------------------------------
def augment_scaling(seq, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(*scale_range)
    return seq * scale

# --------------------------------------------------
# 3) Small random rotation
# Only rotates (x,y) pairs, not the entire 438-vector blindly.
# --------------------------------------------------
def augment_rotation(seq, angle_range=(-5, 5)):
    angle = np.radians(np.random.uniform(*angle_range))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    seq_rot = seq.copy()
    reshaped = seq.reshape(seq.shape[0], -1, 2)

    for t in range(len(reshaped)):
        x = reshaped[t][:, 0]
        y = reshaped[t][:, 1]
        reshaped[t][:, 0] = x * cos_a - y * sin_a
        reshaped[t][:, 1] = x * sin_a + y * cos_a

    return reshaped.reshape(seq.shape)

# --------------------------------------------------
# 4) Time warping by interpolation
# --------------------------------------------------
def augment_time_warp(seq, speed_range=(0.9, 1.1)):
    T = seq.shape[0]
    speed = np.random.uniform(*speed_range)

    # New number of frames
    new_T = int(T * speed)
    new_T = max(5, new_T)

    indices = np.linspace(0, T - 1, new_T)
    warped = np.zeros((new_T, seq.shape[1]))

    for i, idx in enumerate(indices):
        warped[i] = seq[int(idx)]
        
    # Resize back to original length
    indices_fixed = np.linspace(0, new_T - 1, T)
    fixed = np.zeros((T, seq.shape[1]))
    for i, idx in enumerate(indices_fixed):
        fixed[i] = warped[int(idx)]

    return fixed

# --------------------------------------------------
# 5) Random frame drop (mild)
# --------------------------------------------------
def augment_frame_drop(seq, drop_rate=0.05):
    seq_aug = seq.copy()
    T = seq.shape[0]

    num_drop = int(T * drop_rate)
    drop_idx = np.random.choice(T, num_drop, replace=False)

    for idx in drop_idx:
        seq_aug[idx] = seq_aug[idx - 1] if idx > 0 else seq_aug[idx]

    return seq_aug


# --------------------------------------------------
# MASTER FUNCTION: Randomly apply augmentations
# --------------------------------------------------
def augment_sequence(seq):
    if np.random.rand() < 0.5:
        seq = augment_jitter(seq)

    if np.random.rand() < 0.3:
        seq = augment_scaling(seq)

    if np.random.rand() < 0.3:
        seq = augment_rotation(seq)

    if np.random.rand() < 0.4:
        seq = augment_time_warp(seq)

    if np.random.rand() < 0.3:
        seq = augment_frame_drop(seq)

    return seq
