import cv2
import numpy as np
import argparse
import os

# ===== ARGUMENT PARSER =====
parser = argparse.ArgumentParser(description="Compare Original, RAW, and PREPROCESSED skeleton videos")
parser.add_argument("--original", type=str, required=True, help="Path to ORIGINAL video")
parser.add_argument("--raw", type=str, required=True, help="Path to RAW skeleton video")
parser.add_argument("--proc", type=str, required=True, help="Path to PREPROCESSED skeleton video")
parser.add_argument("--delay", type=int, default=100, help="Delay between frames in ms")
args = parser.parse_args()

original_path = args.original
raw_path = args.raw
proc_path = args.proc
delay = args.delay

# ===== VIDEO CAPTURES =====
cap_orig = cv2.VideoCapture(original_path)
cap_raw  = cv2.VideoCapture(raw_path)
cap_proc = cv2.VideoCapture(proc_path)

for cap, name in zip([cap_orig, cap_raw, cap_proc], ["ORIGINAL", "RAW", "PREPROCESSED"]):
    if not cap.isOpened():
        print(f"Failed to open {name} video: {cap}")
        exit(1)

# ===== FIXED CANVAS (CENTER AND SCALE) =====
W, H = 480, 480  # each panel size
def center_on_canvas(frame):
    h, w = frame.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    scale = min(W/w, H/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (new_w, new_h))
    cx, cy = (W - new_w)//2, (H - new_h)//2
    canvas[cy:cy+new_h, cx:cx+new_w] = resized
    return canvas

# ===== SYNC VIDEO LENGTH =====
min_frames = int(min(
    cap_orig.get(cv2.CAP_PROP_FRAME_COUNT),
    cap_raw.get(cv2.CAP_PROP_FRAME_COUNT),
    cap_proc.get(cv2.CAP_PROP_FRAME_COUNT)
))

frame_id = 0
while frame_id < min_frames:
    ret_o, f_o = cap_orig.read()
    ret_r, f_r = cap_raw.read()
    ret_p, f_p = cap_proc.read()
    
    if not (ret_o and ret_r and ret_p):
        break

    f_o = center_on_canvas(f_o)
    f_r = center_on_canvas(f_r)
    f_p = center_on_canvas(f_p)

    # Add labels
    cv2.putText(f_o, "ORIGINAL", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(f_r, "RAW", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(f_p, "PREPROCESSED", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    combined = cv2.hconcat([f_o, f_r, f_p])
    cv2.imshow("Comparison", combined)

    key = cv2.waitKey(delay)
    if key == 27:  # ESC
        break

    frame_id += 1

cap_orig.release()
cap_raw.release()
cap_proc.release()
cv2.destroyAllWindows()