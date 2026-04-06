import os
import numpy as np
import cv2
import argparse

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser(description="Render skeleton videos from .npy sequences")
parser.add_argument("--input", type=str, required=True, help="Path to folder with .npy sequences")
parser.add_argument("--output", type=str, required=True, help="Path to save rendered videos")
args = parser.parse_args()

DATA_ROOT = args.input
OUTPUT_ROOT = args.output

# ===== RENDER FUNCTION =====
def render_skeleton(data, canvas_size=(480,640), dynamic=True):
    h, w = canvas_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # ---- RESHAPE LANDMARKS ----
    pose = data[0:132].reshape(33, 4)
    face = data[132:312].reshape(-1, 3)
    lh   = data[312:375].reshape(21, 3)
    rh   = data[375:438].reshape(21, 3)

    # ---- COLLECT VALID POINTS ----
    points = []
    for p in pose:
        if p[3] > 0.5 and not np.any(np.isnan(p)):
            points.append(p[:2])
    for hand in [lh, rh]:
        for p in hand:
            if not np.any(np.isnan(p)):
                points.append(p[:2])
    for p in face:
        if not np.any(np.isnan(p)):
            points.append(p[:2])
    points = np.array(points)

    # ---- DYNAMIC SCALING ----
    if dynamic and len(points) > 0:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        center = (min_xy + max_xy) / 2
        size = (max_xy - min_xy).max()
        if size < 1e-6: size = 1.0
        scale = 300 / size
    else:
        center = np.array([0,0])
        scale = 200

    def to_pixel(coord):
        x = int((coord[0] - center[0]) * scale + w//2)
        y = int((coord[1] - center[1]) * scale + h//2)
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        return x, y

    # ---- POSE ----
    pose_connections = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),(11,13),(13,15),(15,17),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
        (27,29),(28,30),(29,31),(30,32)
    ]
    for p1, p2 in pose_connections:
        if pose[p1][3] > 0.5 and pose[p2][3] > 0.5:
            x1, y1 = to_pixel(pose[p1])
            x2, y2 = to_pixel(pose[p2])
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    for p in pose:
        if p[3] > 0.5 and not np.any(np.isnan(p)):
            x, y = to_pixel(p)
            cv2.circle(img, (x,y), 3, (0,255,0), -1)

    # ---- HANDS ----
    hand_connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]
    def draw_hand(hand, color):
        for p1,p2 in hand_connections:
            if np.any(np.isnan(hand[p1])) or np.any(np.isnan(hand[p2])):
                continue
            x1,y1 = to_pixel(hand[p1])
            x2,y2 = to_pixel(hand[p2])
            cv2.line(img,(x1,y1),(x2,y2),color,2)
        for p in hand:
            if not np.any(np.isnan(p)):
                x,y = to_pixel(p)
                cv2.circle(img,(x,y),2,color,-1)
    draw_hand(lh,(255,0,0))
    draw_hand(rh,(0,0,255))

    # ---- FACE ----
    for p in face:
        if not np.any(np.isnan(p)):
            x,y = to_pixel(p)
            cv2.circle(img,(x,y),1,(255,255,255),-1)

    return img

# ===== MAIN LOOP =====
for root, dirs, files in os.walk(DATA_ROOT):
    rel_path = os.path.relpath(root, DATA_ROOT)
    output_dir = os.path.join(OUTPUT_ROOT, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    for f in files:
        if not f.endswith(".npy"):
            continue
        npy_path = os.path.join(root, f)
        print(f"Rendering {os.path.join(rel_path,f)}")
        sequence = np.load(npy_path)
        out_path = os.path.join(output_dir, f.replace(".npy",".mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(out_path, fourcc, 30, (640,480))
        if not out.isOpened():
            print(f"Failed to open video writer for {out_path}")
            continue

        for frame in sequence:
            img = render_skeleton(frame, dynamic=True)
            out.write(img)
        out.release()
        print(f"Saved video: {out_path}")

print("All videos rendered and saved!")