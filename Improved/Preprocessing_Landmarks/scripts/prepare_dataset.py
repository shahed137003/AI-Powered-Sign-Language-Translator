import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -----------------------
# CONFIG
# -----------------------
INPUT_DIR = r"D:\Additionalpreprocessing\preprocessed_enhanced_landmarks"
OUTPUT_DIR = r"D:\Additionalpreprocessing\balanced_dataset"

TARGET_SAMPLES_PER_CLASS = 75  
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def extract_label(filename: str) -> str:
    name = Path(filename).stem
    parts = name.split()
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts).strip().upper()

def augment_sample(x, mask):
    # Augmentation functions (Noise, Shift, Warp)
    x_aug, m_aug = x.copy(), mask.copy()
    # 1. Noise
    if random.random() < 0.7:
        x_aug += np.random.normal(0, 0.01, size=x.shape)
    # 2. Shift
    if random.random() < 0.5:
        x_aug = np.roll(x_aug, np.random.randint(-5, 5), axis=0)
    # 3. Warp
    if random.random() < 0.5:
        T, D = x.shape
        factor = np.random.uniform(0.9, 1.1)
        new_T = int(T * factor)
        idx = np.clip(np.linspace(0, T - 1, new_T).astype(int), 0, T - 1)
        x_new = x[idx]
        x_aug = np.concatenate([x_new, np.zeros((T - new_T, D))], axis=0) if new_T < T else x_new[:T]
    return x_aug, m_aug

def main():
    print("🚀 Step 1: Loading Raw Data...")
    all_samples = []
    files = list(Path(INPUT_DIR).glob("*.npy"))
    for file in tqdm(files):
        if "_mask" in file.name: continue
        label = extract_label(file.name)
        try:
            all_samples.append({'x': np.load(file), 'm': np.load(file.with_name(file.stem + "_mask.npy")), 'label': label})
        except: continue

    unique_labels = sorted(list(set(s['label'] for s in all_samples)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    
    # --- FIXED SPLITTING LOGIC ---
    print(f"🚀 Step 2: Splitting Data safely (Total: {len(all_samples)})...")
    
    samples_by_class = defaultdict(list)
    for i, s in enumerate(all_samples):
        samples_by_class[label_to_idx[s['label']]].append(i)

    idx_train, idx_val, idx_test = [], [], []

    for cls_idx, indices in samples_by_class.items():
        n_samples = len(indices)
        
        if n_samples < 3:
            # If class is too small, put everything in training to avoid crash
            idx_train.extend(indices)
        else:
            # Split normally with stratification-like behavior per class
            tr, temp = train_test_split(indices, test_size=(1-TRAIN_RATIO), random_state=SEED)
            
            # Divide temp into val and test
            rel_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
            v, te = train_test_split(temp, test_size=(1-rel_val), random_state=SEED)
            
            idx_train.extend(tr)
            idx_val.extend(v)
            idx_test.extend(te)

    def process_and_save(indices, name, augment=False):
        final_X, final_y, final_m = [], [], []
        
        # Re-group by class for augmentation
        current_split_by_class = defaultdict(list)
        for i in indices:
            s = all_samples[i]
            current_split_by_class[label_to_idx[s['label']]].append(s)

        for cls_idx in range(len(unique_labels)):
            samples = current_split_by_class[cls_idx]
            if not samples: continue

            if augment:
                # Augment up to TARGET_SAMPLES_PER_CLASS
                balanced = samples.copy()
                while len(balanced) < TARGET_SAMPLES_PER_CLASS:
                    source = random.choice(samples)
                    x_a, m_a = augment_sample(source['x'], source['m'])
                    balanced.append({'x': x_a, 'm': m_a})
                if len(balanced) > TARGET_SAMPLES_PER_CLASS:
                    balanced = random.sample(balanced, TARGET_SAMPLES_PER_CLASS)
                samples_to_save = balanced
            else:
                samples_to_save = samples

            for s in samples_to_save:
                final_X.append(s['x'])
                final_m.append(s['m'])
                final_y.append(cls_idx)

        out_path = Path(OUTPUT_DIR) / name
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "X.npy", np.array(final_X))
        np.save(out_path / "y.npy", np.array(final_y))
        np.save(out_path / "mask.npy", np.array(final_m))
        print(f"✅ Saved {name}: {len(final_X)} samples")

    process_and_save(idx_train, "train", augment=True)
    process_and_save(idx_val, "val", augment=False)
    process_and_save(idx_test, "test", augment=False)
    
    np.save(Path(OUTPUT_DIR) / "class_map.npy", label_to_idx)
    print(f"\n🎉 Success! Data split and balanced in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()