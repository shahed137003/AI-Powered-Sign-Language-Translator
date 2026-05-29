import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# =========================
# CONFIG
# =========================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-dir",
    type=str,
    required=True,
    help="Path to input directory"
)

parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Path to output directory"
)

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

MIN_SAMPLES_THRESHOLD = 15

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

#POSE_SIZE = 33 * 4
#FACE_SIZE = 468 * 3
#HAND_SIZE = 21 * 3


# =========================
# LABEL
# =========================
def extract_label(filename: str):
    name = Path(filename).stem

    # remove only the LAST numeric token
    parts = name.rsplit(" ", 1)

    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]

    return name.strip().upper()




# =========================
# SAVE STREAM (NO RAM STORAGE)
# =========================
def save_split(split, name, label_to_idx):
    out_dir = Path(OUTPUT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(tqdm(split)):
        np.savez_compressed(
            out_dir / f"{i}.npz",
            x=s["x"],
            y=label_to_idx[s["label"]]
        )


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")

    all_samples = []
    files = list(Path(INPUT_DIR).glob("*.npy"))

    for f in tqdm(files):


        try:
            x = np.load(f).astype(np.float32)
            label = extract_label(f.name)

            all_samples.append({
                "x": x,
                "label": label
            })

        except:
            continue

    print(f"Total samples: {len(all_samples)}")

    # =========================
    # GROUP BY CLASS
    # =========================
    by_class = defaultdict(list)

    for s in all_samples:
        by_class[s["label"]].append(s)

    by_class = {
        k: v for k, v in by_class.items()
        if len(v) >= MIN_SAMPLES_THRESHOLD
    }

    labels = sorted(by_class.keys())
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # =========================
    # SPLIT (NO LEAK)
    # =========================
    train, val, test = [], [], []

    for label, samples in by_class.items():

        tr, temp = train_test_split(
            samples,
            test_size=(1 - TRAIN_RATIO),
            random_state=SEED
        )

        rel_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

        v, te = train_test_split(
            temp,
            test_size=(1 - rel_val),
            random_state=SEED
        )

        train += tr
        val += v
        test += te

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    

    # =========================
    # STREAM SAVE
    # =========================
    print("Saving train...")
    save_split(
    train,
    "train",
    label_to_idx
)

    print("Saving val...")
    save_split(
    val,
    "val",
    label_to_idx
)

    print("Saving test...")
    
    save_split(
    test,
    "test",
    label_to_idx
)

    np.save(Path(OUTPUT_DIR) / "label_encoder.npy", label_to_idx)

    print("Done")


if __name__ == "__main__":
    main()