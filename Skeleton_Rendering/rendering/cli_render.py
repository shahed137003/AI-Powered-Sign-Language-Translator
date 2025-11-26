import os
import argparse
import numpy as np
import cv2

from .layout_438 import FEATURE_DIM, unpack_frame
from .connections import pose_connections, hand_connections
from .bbox import compute_bbox_for_clip
from .mapper import make_mapper
from .draw import draw_pose, draw_hand, draw_face


# ---------- main rendering ----------
def main():
    ap = argparse.ArgumentParser(
        description="Render skeleton videos for arbitrary (root+scaled) coordinates."
    )
    ap.add_argument("--input-dir", type=str, required=True,
                    help="Directory with .npy keypoint files (T x 438)")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Directory to save .avi videos")
    ap.add_argument("--limit", type=int, default=10,
                    help="Render at most N files (0 = all)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--fourcc", type=str, default="XVID")
    ap.add_argument("--pose-vis-thresh", type=float, default=0.0,
                    help="Skip pose connections if vis <= this (after preprocessing)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npy")])

    if args.limit and args.limit > 0:
        files = files[:args.limit]

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        sample = np.load(in_path, allow_pickle=True)

        if sample.ndim != 2 or sample.shape[1] != FEATURE_DIM:
            print(f"Skipping {fname}: expected (T,{FEATURE_DIM}), got {sample.shape}")
            continue

        xmin, xmax, ymin, ymax = compute_bbox_for_clip(sample)
        map_xy = make_mapper(xmin, xmax, ymin, ymax, args.width, args.height)

        out_name = fname.replace(".npy", "_robust_skel.avi")
        out_path = os.path.join(args.output_dir, out_name)
        vw = cv2.VideoWriter(out_path, fourcc, args.fps, (args.width, args.height))

        for t in range(sample.shape[0]):
            pose, face, lh, rh = unpack_frame(sample[t])
            img = np.ones((args.height, args.width, 3), np.uint8) * 255

            draw_pose(
                img=img,
                pose=pose,
                map_xy=map_xy,
                pose_connections=pose_connections,
                pose_vis_thresh=args.pose_vis_thresh,
                color=(0, 255, 0),
                thickness=2,
            )

            draw_hand(
                img=img,
                hand=lh,
                map_xy=map_xy,
                hand_connections=hand_connections,
                color=(255, 0, 0),
                thickness=2,
            )

            draw_hand(
                img=img,
                hand=rh,
                map_xy=map_xy,
                hand_connections=hand_connections,
                color=(0, 0, 255),
                thickness=2,
            )

            draw_face(
                img=img,
                face=face,
                map_xy=map_xy,
                color=(255, 0, 255),
                radius=2,
            )

            vw.write(img)

        vw.release()
        print("Saved:", out_path)

    print("Done. Rendered:", len(files), "videos to", args.output_dir)


if __name__ == "__main__":
    main()
