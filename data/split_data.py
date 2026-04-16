import shutil
import csv
from pathlib import Path
from collections import defaultdict
import argparse
import random

# ARGUMENTS 
parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
parser.add_argument("input_dir", type=str, help="Path to input directory containing .npy files")
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = INPUT_DIR.parent / "splits"
CSV_PATH = OUTPUT_DIR / "splits.csv"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_label(path: Path):
    name = path.stem.strip()
    parts = name.rsplit(" ", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].strip()

    return name


# GROUP FILES
groups = defaultdict(list)

for f in INPUT_DIR.rglob("*.npy"):
    label = get_label(f)
    groups[label].append(f)



# REMOVE SMALL CLASSES (<10)
MIN_SAMPLES = 10

filtered_groups = {}

for label, files in groups.items():
    if len(files) < MIN_SAMPLES:
        print(f"Gloss '{label}' removed because has {len(files)} samples")
    else:
        filtered_groups[label] = files

groups = filtered_groups


# SPLIT FUNCTION
def split_group(files):
    files = sorted(files)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    return train, val, test


# COPY + RECORD
def copy_and_record(file_list, split_name, rows):
    for f in file_list:
        out_path = OUTPUT_DIR / split_name / f.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(f, out_path)

        rows.append({
            "filepath": str(out_path.relative_to(OUTPUT_DIR)),
            "label": get_label(f),
            "split": split_name
        })


# MAIN
rows = []

total_train = total_val = total_test = 0

for label, files in groups.items():
    train, val, test = split_group(files)

    copy_and_record(train, "train", rows)
    copy_and_record(val, "val", rows)
    copy_and_record(test, "test", rows)

    total_train += len(train)
    total_val += len(val)
    total_test += len(test)


# SAVE CSV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
    writer.writeheader()
    writer.writerows(rows)


print("Done!")
print(f"Train samples: {total_train}")
print(f"Val samples: {total_val}")
print(f"Test samples: {total_test}")
print(f"Total classes: {len(groups)}")
print(f"CSV saved to: {CSV_PATH}")