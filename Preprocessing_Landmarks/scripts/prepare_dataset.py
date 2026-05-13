import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
INPUT_DIR = r"E:\500 new with cleaning\165_preprocessed_with_features"
OUTPUT_DIR = r"E:\500 new with cleaning\165_final_dataset"

MIN_SAMPLES_THRESHOLD = 15
MIN_TARGET_PER_CLASS = 45

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
    parts = name.split()
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts).strip().upper()


# =========================
# LOAD MASK (SAFE)
# =========================
def load_mask(file_path):
    mask_path = str(file_path).replace(".npy", "_mask.npy")
    if Path(mask_path).exists():
        try:
            return np.load(mask_path).astype(np.float32)
        except:
            return None
    return None

def augment_sample(x):
    T, D = x.shape

    out = x.copy()

    # -----------------------
    # 1. SPARSE NOISE
    # -----------------------
    if random.random() < 0.6:
        idx = np.random.randint(0, T, size=max(2, T // 10))

        noise = np.random.normal(
            0,
            0.003,
            (len(idx), D)
        ).astype(np.float32)

        out[idx] += noise

    # -----------------------
    # 2. TEMPORAL SHIFT
    # -----------------------
    if random.random() < 0.4:
        shift = random.randint(-2, 2)

        if shift != 0:
            tmp = np.zeros_like(out)

            if shift > 0:
                tmp[shift:] = out[:-shift]
            else:
                tmp[:shift] = out[-shift:]

            out = tmp

    # -----------------------
    # 3. SPEED VARIATION
    # -----------------------
    if random.random() < 0.4:
        factor = random.uniform(0.9, 1.1)

        idx = np.linspace(
            0,
            T - 1,
            max(2, int(T * factor))
        ).astype(np.int32)

        idx = np.clip(idx, 0, T - 1)

        temp = out[idx]

        if temp.shape[0] > T:
            out = temp[:T]

        else:
            pad = np.repeat(
                temp[-1:],
                T - temp.shape[0],
                axis=0
            )

            out = np.vstack([temp, pad])

    # -----------------------
    # 4. FEATURE DROPOUT
    # -----------------------
    if random.random() < 0.3:
        cols = np.random.randint(
            0,
            D,
            size=max(1, D // 120)
        )

        out[:, cols] = 0.0

    # -----------------------
    # 5. LIGHT GAUSSIAN NOISE
    # -----------------------
    if random.random() < 0.3:
        idx = np.random.randint(
            0,
            T,
            size=max(2, T // 20)
        )

        noise = np.random.normal(
            0,
            0.001,
            (len(idx), D)
        ).astype(np.float32)

        out[idx] += noise

    # -----------------------
    # 6. FRAME DROPOUT
    # -----------------------
    if random.random() < 0.2:
        n_drop = max(1, T // 20)

        drop_idx = np.random.choice(
            T,
            n_drop,
            replace=False
        )

        out[drop_idx] = 0.0

    return out.astype(np.float32)


# =========================
# SAVE STREAM (NO RAM STORAGE)
# =========================
def save_split(split, name, label_to_idx, augment=False):
    out_dir = Path(OUTPUT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(tqdm(split)):
        x = s["x"].copy()

        if augment:
            x = augment_sample(x)

        np.savez_compressed(
            out_dir / f"{i}.npz",
            x=x,
            y=label_to_idx[s["label"]],
            mask=s["mask"]   # <<< INCLUDED SAFELY
        )


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")

    all_samples = []
    files = list(Path(INPUT_DIR).glob("*.npy"))

    for f in tqdm(files):
        if "_mask" in f.name:
            continue

        try:
            x = np.load(f).astype(np.float32)
            label = extract_label(f.name)
            mask = load_mask(f)

            all_samples.append({
                "x": x,
                "mask": mask,
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
    # BALANCE TRAIN ONLY (NO AUG HERE)
    # =========================
    train_by_class = defaultdict(list)

    for s in train:
        train_by_class[s["label"]].append(s)

    max_size = max(len(v) for v in train_by_class.values())
    target = max(max_size, MIN_TARGET_PER_CLASS)

    balanced_train = []

    for label, samples in train_by_class.items():
        pool = list(samples)

        while len(pool) < target:
            pool.append(random.choice(samples))

        balanced_train.extend(random.sample(pool, target))

    print(f"Balanced train: {len(balanced_train)}")

    # =========================
    # STREAM SAVE
    # =========================
    print("Saving train...")
    save_split(
        balanced_train,
        "train",
        label_to_idx,
        augment=True
    )

    print("Saving val...")
    save_split(
        val,
        "val",
        label_to_idx,
        augment=False
    )

    print("Saving test...")
    
    save_split(
        test,
        "test",
        label_to_idx,
        augment=False
    )

    np.save(Path(OUTPUT_DIR) / "class_map.npy", label_to_idx)

    print("Done")


if __name__ == "__main__":
    main()