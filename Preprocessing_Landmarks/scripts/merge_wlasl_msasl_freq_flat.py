#!/usr/bin/env python
"""Merge preprocessed WLASL and MS-ASL keypoint datasets into a frequency-filtered
dataset with train/val/test splits, assuming a FLAT directory structure.

Expected structure (and ONLY this structure):

    WLASL root:
        <root>/*.npy

    MS-ASL root:
        <root>/*.npy

File naming convention:

    <label>.npy
    <label>_<anything>.npy

The label is taken as everything before the first underscore `_` in the filename
stem. If there is no underscore, the entire stem is used as the label.

The script will:
- collect all .npy files from WLASL and MS-ASL roots,
- infer a label (gloss) for each file from its filename,
- compute total samples per label (wlasl + msasl),
- keep only labels with total >= --min-total,
- shuffle files per label and split into train/val/test,
- copy them into an output directory structured as:

    <out-root>/
        train/<label>/*.npy
        val/<label>/*.npy
        test/<label>/*.npy

Usage example:

    python merge_wlasl_msasl_freq.py \\
        --wlasl-root "C:\\...\\WLASL_Keypoints_preproc" \\
        --msasl-root "C:\\...\\MSASL_Keypoints_preproc" \\
        --out-root  "C:\\...\\MERGED_FREQ20" \\
        --min-total 20
"""

import argparse
from pathlib import Path
import random
import shutil
from typing import Dict, List


def infer_label_from_filename(f: Path) -> str:
    """Infer a label (gloss) from a flat .npy filename.

    Rules (for this project only):
    - If filename is 'your_UoITyyziLOw.npy' -> label = 'your'
    - If filename is 'zero.npy'             -> label = 'zero'
    """
    stem = f.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def collect_files_flat(root: Path) -> Dict[str, List[Path]]:
    """Collect all .npy files directly under *root* and group them by label.

    Assumes a FLAT directory:
        root/*.npy

    Returns
    -------
    dict
        Mapping label -> list of Paths to .npy files.
    """
    label_to_files: Dict[str, List[Path]] = {}
    npy_files = sorted(root.glob("*.npy"))
    print(f"[{root}] found {len(npy_files)} .npy files")  # noqa: T201

    for f in npy_files:
        lab = infer_label_from_filename(f)
        label_to_files.setdefault(lab, []).append(f)

    print(f"[{root}] labels: {len(label_to_files)}")  # noqa: T201
    return label_to_files


def ensure_dir(p: Path) -> None:
    """Create directory *p* if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description=(
            "Merge preprocessed WLASL and MS-ASL keypoint datasets into a single "
            "frequency-filtered dataset with train/val/test splits, assuming a "
            "flat root/*.npy structure."
        )
    )
    ap.add_argument(
        "--wlasl-root",
        type=str,
        required=True,
        help=(
            "Root of WLASL preprocessed keypoints, e.g. "
            "'WLASL_Keypoints_preproc_v3_all' (flat dir with .npy files)."
        ),
    )
    ap.add_argument(
        "--msasl-root",
        type=str,
        required=True,
        help=(
            "Root of MS-ASL preprocessed keypoints, e.g. "
            "'MSASL_Keypoints_preproc_v3_all' (flat dir with .npy files)."
        ),
    )
    ap.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Output root for merged dataset, e.g. 'MERGED_FREQ20'. "
            "Will contain train/val/test subdirectories."
        ),
    )
    ap.add_argument(
        "--min-total",
        type=int,
        default=20,
        help=(
            "Minimum total samples (wlasl + msasl) per label to keep. "
            "Labels with fewer samples are discarded (default: 20)."
        ),
    )
    ap.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of samples to allocate to the training split (default: 0.8)."
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of samples to allocate to the validation split (default: 0.1)."
    )
    ap.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of samples to allocate to the test split (default: 0.1)."
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for shuffling files before splitting (default: 123)."
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(
            f"train + val + test ratios must sum to 1.0, got {ratio_sum:.6f}."
        )

    if args.min_total <= 0:
        raise SystemExit("--min-total must be a positive integer.")

    random.seed(args.seed)

    w_root = Path(args.wlasl_root).expanduser().resolve()
    m_root = Path(args.msasl_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not w_root.is_dir():
        raise SystemExit(f"WLASL root does not exist or is not a directory: {w_root}")
    if not m_root.is_dir():
        raise SystemExit(f"MS-ASL root does not exist or is not a directory: {m_root}")

    print("Collecting WLASL files (flat dir)...")  # noqa: T201
    w_files = collect_files_flat(w_root)
    print("Collecting MS-ASL files (flat dir)...")  # noqa: T201
    m_files = collect_files_flat(m_root)

    # Merge counts across datasets
    all_labels = set(w_files.keys()) | set(m_files.keys())
    total_counts: Dict[str, int] = {}
    for lab in all_labels:
        cw = len(w_files.get(lab, []))
        cm = len(m_files.get(lab, []))
        total_counts[lab] = cw + cm

    # Filter labels by min_total
    keep_labels = [lab for lab, c in total_counts.items() if c >= args.min_total]
    keep_labels = sorted(keep_labels)
    print(f"Total labels (union): {len(all_labels)}")  # noqa: T201
    print(  # noqa: T201
        f"Keeping {len(keep_labels)} labels with total >= {args.min_total}"
    )

    if not keep_labels:
        print("No labels satisfy min_total. Exiting.")  # noqa: T201
        return

    # Create output structure: out_root/{train,val,test}
    ensure_dir(out_root)
    for split in ("train", "val", "test"):
        ensure_dir(out_root / split)

    # For each kept label: gather files from both datasets, shuffle, split, copy
    for lab in keep_labels:
        src_files: List[Path] = []
        src_files.extend(w_files.get(lab, []))
        src_files.extend(m_files.get(lab, []))

        if not src_files:
            continue

        random.shuffle(src_files)
        n = len(src_files)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        # ensure sum == n
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        print(  # noqa: T201
            f"Label '{lab}': total={n} -> train={n_train}, val={n_val}, test={n_test}"
        )

        # Create label dirs
        train_dir = out_root / "train" / lab
        val_dir = out_root / "val" / lab
        test_dir = out_root / "test" / lab
        ensure_dir(train_dir)
        ensure_dir(val_dir)
        ensure_dir(test_dir)

        # Assign files
        train_files = src_files[:n_train]
        val_files = src_files[n_train:n_train + n_val]
        test_files = src_files[n_train + n_val:]

        # Copy with unique names (prefix split to avoid collisions)
        for idx, f in enumerate(train_files):
            dst = train_dir / f"{lab}_train_{idx:05d}.npy"
            shutil.copy2(f, dst)

        for idx, f in enumerate(val_files):
            dst = val_dir / f"{lab}_val_{idx:05d}.npy"
            shutil.copy2(f, dst)

        for idx, f in enumerate(test_files):
            dst = test_dir / f"{lab}_test_{idx:05d}.npy"
            shutil.copy2(f, dst)

    print("Done. Merged dataset written to:", out_root)  # noqa: T201


if __name__ == "__main__":
    main()
