import cv2
import os
import argparse
import numpy as np

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser(description="Batch Compare Videos Without Overlay Artifacts")
parser.add_argument("--original_dir", type=str, required=True)
parser.add_argument("--raw_dir", type=str, required=True)
parser.add_argument("--proc_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--delay", type=int, default=1)
args = parser.parse_args()

# ===== SETTINGS =====
W, H = 480, 480  # size of each panel

def center_on_canvas(frame):
    """Resize and center a frame on fixed canvas"""
    h, w = frame.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)  # fresh canvas every frame
    scale = min(W / w, H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    cx, cy = (W - new_w) // 2, (H - new_h) // 2
    canvas[cy:cy+new_h, cx:cx+new_w] = resized
    return canvas

# ===== PROCESS ALL FILES =====
for root, _, files in os.walk(args.raw_dir):
    for file in files:
        if not file.endswith(".mp4"):
            continue

        raw_path = os.path.join(root, file)
        rel_path = os.path.relpath(raw_path, args.raw_dir)
        original_path = os.path.join(args.original_dir, rel_path)
        proc_path     = os.path.join(args.proc_dir, rel_path)
        output_path   = os.path.join(args.output_dir, rel_path)

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\nProcessing: {file}")

        # ===== OPEN VIDEOS =====
        cap_orig = cv2.VideoCapture(original_path)
        cap_raw  = cv2.VideoCapture(raw_path)
        cap_proc = cv2.VideoCapture(proc_path)

        if not (cap_orig.isOpened() and cap_raw.isOpened() and cap_proc.isOpened()):
            print("❌ Skipping (missing video)")
            continue

        # ===== VIDEO WRITER =====
        fps = cap_raw.get(cv2.CAP_PROP_FPS) or 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W * 3, H))

        # ===== FRAME LOOP =====
        while True:
            ret_o, f_o = cap_orig.read()
            ret_r, f_r = cap_raw.read()
            ret_p, f_p = cap_proc.read()
            if not (ret_o and ret_r and ret_p):
                break

            # Create fresh canvas for each frame (avoids extra landmarks)
            f_o_canvas = center_on_canvas(f_o)
            f_r_canvas = center_on_canvas(f_r)
            f_p_canvas = center_on_canvas(f_p)

            # Labels
            cv2.putText(f_o_canvas, "ORIGINAL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(f_r_canvas, "RAW", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(f_p_canvas, "PREPROCESSED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            combined = cv2.hconcat([f_o_canvas, f_r_canvas, f_p_canvas])
            out.write(combined)

        cap_orig.release()
        cap_raw.release()
        cap_proc.release()
        out.release()

        print(f"✅ Saved: {output_path}")

print("\nAll videos processed!")