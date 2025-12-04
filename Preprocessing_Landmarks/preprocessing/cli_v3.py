from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .constants import FEATURE_DIM
from .io_utils import iter_npy_files, load_keypoints_npy, save_npy
from .pipeline_v3 import preprocess_sequence_global


def fix_length_to_target(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ensure sequence has exactly target_len frames on axis 0 by trimming or padding.

    - If shorter: pad by repeating the last frame.
    - If longer: trim to the first target_len frames.
    """
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected shape (T,{FEATURE_DIM}), got {x.shape}")

    t = x.shape[0]
    if t == target_len:
        return x
    if t > target_len:
        return x[:target_len]

    pad_len = target_len - t
    last = x[-1:, :]
    pad = np.repeat(last, pad_len, axis=0)
    return np.concatenate([x, pad], axis=0)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="v3: global-mean root+scale + keepwrists + tiered hand fill + optional OneEuro smoothing."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-npy", type=str, help="Path to one .npy file (shape T x 438)")
    g.add_argument("--input-dir", type=str, help="Directory containing .npy files (recursive)")

    ap.add_argument("--output-npy", type=str, help="Output path for one-file mode")
    ap.add_argument("--output-dir", type=str, help="Output directory for directory mode")

    ap.add_argument("--pose-vis-thresh", type=float, default=0.1,
                    help="Pose vis threshold (default: 0.1). Critical joints are kept for transform.")
    ap.add_argument("--keep-legs", action="store_true",
                    help="If set, do NOT zero leg joints (25..32)")

    ap.add_argument("--no-fix-swap", action="store_true",
                    help="Disable swap-fix and wrist-distance gating (default: enabled)")
    ap.add_argument("--no-fill-hands", action="store_true",
                    help="Disable hand gap fill (default: enabled)")

    ap.add_argument("--max-gap", type=int, default=6,
                    help="Small gaps: fill up to this many frames (default: 6)")
    ap.add_argument("--medium-gap", type=int, default=15,
                    help="Medium gaps: wrist-follow static carry up to this many frames (default: 15)")

    ap.add_argument("--min-hand-pts", type=int, default=8,
                    help="Hand present if >= this many non-zero landmarks (default: 8)")
    ap.add_argument("--hand-wrist-max-dist", type=float, default=1.1,
                    help="If hand centroid farther than this from wrist, zero it. (default: 1.1)")
    ap.add_argument("--rel-change-thresh", type=float, default=0.7,
                    help="Small gaps: if wrist-relative shape changes > this, carry-forward instead of interp. (default: 0.7)")

    # smoothing
    ap.add_argument("--smooth", action="store_true", help="Enable OneEuro smoothing (default: off)")
    ap.add_argument("--smooth-fps", type=float, default=20.0, help="FPS for smoothing (default: 20)")
    ap.add_argument("--no-smooth-pose", action="store_true", help="Disable smoothing pose")
    ap.add_argument("--no-smooth-hands", action="store_true", help="Disable smoothing hands")
    ap.add_argument("--smooth-face", action="store_true", help="Enable smoothing face (default: off)")

    ap.add_argument("--pose-min-cutoff", type=float, default=1.5)
    ap.add_argument("--pose-beta", type=float, default=0.6)
    ap.add_argument("--hand-min-cutoff", type=float, default=3.0)
    ap.add_argument("--hand-beta", type=float, default=0.8)
    ap.add_argument("--face-min-cutoff", type=float, default=2.0)
    ap.add_argument("--face-beta", type=float, default=0.6)
    ap.add_argument("--d-cutoff", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=0,
                    help="In dir mode, process at most N files (0 = all)")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle file order before processing (default: off)")
    ap.add_argument("--seed", type=int, default=123,
                    help="Seed used only if --shuffle is set")

    ap.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help="If set, trim or pad each sequence on time axis to this many frames (e.g., 96)."
    )

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    def run_one(x: np.ndarray) -> np.ndarray:
        return preprocess_sequence_global(
            x,
            pose_vis_thresh=args.pose_vis_thresh,
            keep_legs=args.keep_legs,
            fix_swap=(not args.no_fix_swap),
            fill_hands=(not args.no_fill_hands),
            small_gap=args.max_gap,
            medium_gap=args.medium_gap,
            min_hand_pts=args.min_hand_pts,
            hand_wrist_max_dist=args.hand_wrist_max_dist,
            rel_change_thresh=args.rel_change_thresh,
            smooth=args.smooth,
            smooth_fps=args.smooth_fps,
            smooth_pose=(not args.no_smooth_pose),
            smooth_hands=(not args.no_smooth_hands),
            smooth_face=args.smooth_face,
            pose_min_cutoff=args.pose_min_cutoff,
            pose_beta=args.pose_beta,
            hand_min_cutoff=args.hand_min_cutoff,
            hand_beta=args.hand_beta,
            face_min_cutoff=args.face_min_cutoff,
            face_beta=args.face_beta,
            d_cutoff=args.d_cutoff,
        )

    # Single-file mode
    if args.input_npy:
        if not args.output_npy:
            raise SystemExit("--output-npy is required when using --input-npy")

        x = np.load(args.input_npy, allow_pickle=True)
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise SystemExit(f"Expected shape (T,{FEATURE_DIM}), got {x.shape}")

        y = run_one(x)
        if args.target_frames is not None:
            y = fix_length_to_target(y, args.target_frames)
        save_npy(Path(args.output_npy), y)
        print("Saved:", args.output_npy)
        return

    # Directory mode
    if not args.output_dir:
        raise SystemExit("--output-dir is required when using --input-dir")

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in iter_npy_files(in_root)])
    if args.shuffle:
        files = list(rng.permutation(files))

    n = 0
    skipped = 0
    for in_path in files:
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            x = load_keypoints_npy(in_path)
        except ValueError as e:
            print(f"Skipping (bad shape): {in_path} ({e})")
            skipped += 1
            continue

        y = run_one(x)
        if args.target_frames is not None:
            try:
                y = fix_length_to_target(y, args.target_frames)
            except ValueError as e:
                print(f"Skipping (bad shape after preprocess): {in_path} ({e})")
                skipped += 1
                continue

        save_npy(out_path, y)

        n += 1
        if n % 200 == 0:
            print("Processed:", n)
        if args.limit and n >= args.limit:
            break

    print("Done. Processed:", n, "Skipped:", skipped)
    print("Output root:", out_root)

