from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .constants import FEATURE_DIM
from .io_utils import iter_npy_files, load_keypoints_npy, save_npy
from .pipeline_v3 import preprocess_sequence_global
from preprocessing.features import build_features


def assert_finite(x, path=None):
    if not np.isfinite(x).all():
        raise ValueError(f"NaN/Inf detected at {path}")


def apply_padding(x: np.ndarray, target_frames: int | None):
    if target_frames is None:
        return x

    if x.shape[0] == target_frames:
        return x

    if x.shape[0] > target_frames:
        return x[:target_frames]

    pad_len = target_frames - x.shape[0]
    pad = np.zeros((pad_len, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)





def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-npy", type=str)
    g.add_argument("--input-dir", type=str)

    ap.add_argument("--output-npy", type=str)
    ap.add_argument("--output-dir", type=str)

    ap.add_argument("--pose-vis-thresh", type=float, default=0.1)
    ap.add_argument("--keep-legs", action="store_true")

    ap.add_argument("--no-fix-swap", action="store_true")
    ap.add_argument("--no-fill-hands", action="store_true")

    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--medium-gap", type=int, default=15)

    ap.add_argument("--min-hand-pts", type=int, default=8)
    ap.add_argument("--hand-wrist-max-dist", type=float, default=1.1)
    ap.add_argument("--rel-change-thresh", type=float, default=0.7)

    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--smooth-fps", type=float, default=20.0)
    ap.add_argument("--no-smooth-pose", action="store_true")
    ap.add_argument("--no-smooth-hands", action="store_true")
    ap.add_argument("--smooth-face", action="store_true")

    ap.add_argument("--pose-min-cutoff", type=float, default=1.5)
    ap.add_argument("--pose-beta", type=float, default=0.6)
    ap.add_argument("--hand-min-cutoff", type=float, default=3.0)
    ap.add_argument("--hand-beta", type=float, default=0.8)
    ap.add_argument("--face-min-cutoff", type=float, default=2.0)
    ap.add_argument("--face-beta", type=float, default=0.6)
    ap.add_argument("--d-cutoff", type=float, default=1.0)

    ap.add_argument("--target-frames", type=int, default=None)

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

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

    # SINGLE FILE
    if args.input_npy:
        if not args.output_npy:
            raise SystemExit("--output-npy is required")

        data = np.load(args.input_npy, allow_pickle=True)
        x = data["x"]
        y = data["y"]

        seq = run_one(x)
        seq = apply_padding(seq, args.target_frames)

        # CREATE MASK BEFORE FEATURE ENGINEERING
        mask = (np.abs(seq).sum(axis=-1) > 0).astype(np.float32)

        features = build_features(seq)

        assert_finite(features)
        out_path = Path(args.output_npy).with_suffix(".npz")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(args.output_npy).with_suffix(".npz"),
            x=features,
            y=y,
            mask=mask
        )
        return

    # DIRECTORY MODE
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for in_path in iter_npy_files(in_root):
        try:
            data = np.load(in_path, allow_pickle=True)
            x = data["x"]
            y = data["y"]
        except:
            continue

        seq = run_one(x)
        seq = apply_padding(seq, args.target_frames)

        # CREATE MASK BEFORE FEATURE ENGINEERING
        mask = (np.abs(seq).sum(axis=-1) > 0).astype(np.float32)

        features = build_features(seq)
        
        out_path = (out_root / in_path.relative_to(in_root)).with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
        out_path,
        x=features,
        y=y,
        mask=mask
    )

    print("Done.")