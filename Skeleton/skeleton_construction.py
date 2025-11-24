import os
import argparse
import numpy as np
import cv2

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(description="Generate skeleton videos from keypoint arrays.")
parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .npy keypoint files")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated videos")
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Collect .npy files
npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

# ----------------------------
# Keypoint layout configuration
# ----------------------------
POSE_LANDMARKS = 33
POSE_VALS = 4
POSE_SIZE = POSE_LANDMARKS * POSE_VALS          # 132

HAND_LANDMARKS = 21
HAND_VALS = 3
HAND_SIZE = HAND_LANDMARKS * HAND_VALS          # 63
HANDS_TOTAL = HAND_SIZE * 2                     # 126

FACE_LANDMARKS = 60
FACE_VALS = 3
FACE_SIZE = FACE_LANDMARKS * FACE_VALS          # 180

FEATURE_DIM = POSE_SIZE + HANDS_TOTAL + FACE_SIZE   # 438


# ----------------------------
# Convert a frame vector into landmark arrays
# ----------------------------
def unpack_frame(vec):
    pose = vec[:POSE_SIZE].reshape(POSE_LANDMARKS, POSE_VALS)
    face = vec[POSE_SIZE:POSE_SIZE + FACE_SIZE].reshape(FACE_LANDMARKS, FACE_VALS)
    lh   = vec[POSE_SIZE + FACE_SIZE : POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(HAND_LANDMARKS, HAND_VALS)
    rh   = vec[POSE_SIZE + FACE_SIZE + HAND_SIZE:].reshape(HAND_LANDMARKS, HAND_VALS)
    return pose, face, lh, rh


# ----------------------------
# Skeleton connections
# ----------------------------
pose_connections = [
    (0,1),(1,2),(2,3), (0,4),(4,5),(5,6),
    (9,10), (7,11), (8,12),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32)
]

hand_connections = [
    (0,1),(1,2),(2,3),(3,4),
    (5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),
]

# ----------------------------
# Render settings
# ----------------------------
W, H = 640, 480


# ----------------------------
# Draw one frame into the video
# ----------------------------
def draw_frame(idx):
    frame = sample[idx]
    pose, face, lh, rh = unpack_frame(frame)

    img = np.ones((H, W, 3), np.uint8) * 255

    # Pose
    for s, e in pose_connections:
        cv2.line(img,
                 (int(pose[s,0]*W), int(pose[s,1]*H)),
                 (int(pose[e,0]*W), int(pose[e,1]*H)),
                 (0,255,0), 2)

    # Left hand
    for s, e in hand_connections:
        cv2.line(img,
                 (int(lh[s,0]*W), int(lh[s,1]*H)),
                 (int(lh[e,0]*W), int(lh[e,1]*H)),
                 (255,0,0), 2)

    # Right hand
    for s, e in hand_connections:
        cv2.line(img,
                 (int(rh[s,0]*W), int(rh[s,1]*H)),
                 (int(rh[e,0]*W), int(rh[e,1]*H)),
                 (0,0,255), 2)

    # Face points
    for i in range(FACE_LANDMARKS):
        x = int(face[i,0] * W)
        y = int(face[i,1] * H)
        cv2.circle(img, (x, y), 2, (255,0,255), -1)

    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ----------------------------
# Process all .npy files
# ----------------------------
for file in npy_files:

    path = os.path.join(input_dir, file)
    sample = np.load(path, allow_pickle=True)

    if sample.shape[1] != FEATURE_DIM:
        print(f"Skipping {file}: expected {FEATURE_DIM} values per frame, got {sample.shape[1]}")
        continue

    out_name = file.replace(".npy", "_skeleton.avi")
    out_path = os.path.join(output_dir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(out_path, fourcc, 20.0, (W, H))

    for i in range(sample.shape[0]):
        draw_frame(i)

    video.release()
    print(f"Saved {out_path}")

print("All videos finished.")
