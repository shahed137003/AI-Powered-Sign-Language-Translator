#!/usr/bin/env python3
# ============================================================
# AUGMENTATION PIPELINE FOR SIGN LANGUAGE LANDMARK DATA
# ============================================================
# Usage: python augment_data.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
# ============================================================

import argparse
import numpy as np
import random
from typing import List, Tuple, Optional
import warnings
import gc
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import pickle

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================

# Feature dimensions (based on 438 feature format)
POSE_SIZE = 132      # 33 landmarks × 4 values
FACE_SIZE = 180      # 60 landmarks × 3 values
HAND_SIZE = 63       # 21 landmarks × 3 values

# Feature indices
POSE_START = 0
POSE_END = POSE_SIZE
FACE_START = POSE_SIZE
FACE_END = POSE_SIZE + FACE_SIZE
LEFT_HAND_START = POSE_SIZE + FACE_SIZE
LEFT_HAND_END = POSE_SIZE + FACE_SIZE + HAND_SIZE
RIGHT_HAND_START = POSE_SIZE + FACE_SIZE + HAND_SIZE
RIGHT_HAND_END = POSE_SIZE + FACE_SIZE + (2 * HAND_SIZE)
FEATURE_DIM = 438


# ============================================================
# DATA VALIDATION AND CONVERSION
# ============================================================

def validate_and_convert_data(data_array):
    """Convert data to proper float32 format"""
    if isinstance(data_array, np.ndarray):
        if data_array.dtype == np.object_:
            print("⚠️ Converting object array to float32...")
            converted = []
            for i in range(len(data_array)):
                if isinstance(data_array[i], np.ndarray):
                    converted.append(data_array[i].astype(np.float32))
                else:
                    converted.append(np.array(data_array[i], dtype=np.float32))
            return np.array(converted, dtype=np.float32)
        elif data_array.dtype != np.float32:
            return data_array.astype(np.float32)
    return data_array


# ============================================================
# MEMORY-EFFICIENT AUGMENTER
# ============================================================

class MemoryEfficientAugmenter:
    """
    Memory-efficient augmentation for sign language landmarks
    - Processes one sample at a time
    - Only modifies hand landmarks (preserves face and pose)
    - Validates all augmentations
    """
    
    def __init__(self, feature_dim: int = FEATURE_DIM):
        self.feature_dim = feature_dim
        
        self.POSE_SIZE = POSE_SIZE
        self.FACE_SIZE = FACE_SIZE
        self.HAND_SIZE = HAND_SIZE
        
        self.POSE_START = POSE_START
        self.POSE_END = POSE_END
        self.FACE_START = FACE_START
        self.FACE_END = FACE_END
        self.LH_START = LEFT_HAND_START
        self.LH_END = LEFT_HAND_END
        self.RH_START = RIGHT_HAND_START
        self.RH_END = RIGHT_HAND_END
        
        self.stats = {'augmented': 0, 'failed': 0}
    
    def _is_valid(self, seq: np.ndarray) -> bool:
        """Quick validation"""
        if seq is None or len(seq) == 0:
            return False
        try:
            if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
                return False
        except:
            return False
        if np.all(seq == 0):
            return False
        return True
    
    def _has_hand(self, hand: np.ndarray, threshold: float = 1e-6) -> bool:
        """Check if hand has any non-zero landmarks"""
        if hand is None or not isinstance(hand, np.ndarray):
            return False
        try:
            return bool(np.any(np.abs(hand) > threshold))
        except:
            return False
    
    def apply_speed_variance(self, sequence: np.ndarray, 
                             speed_factor: float = None,
                             mode: str = 'balanced') -> np.ndarray:
        """Apply speed variance using frame sampling"""
        if speed_factor is None:
            if mode == 'faster':
                speed_factor = random.uniform(1.05, 1.25)
            elif mode == 'slower':
                speed_factor = random.uniform(0.75, 0.95)
            elif mode == 'balanced':
                if random.random() < 0.5:
                    speed_factor = random.uniform(1.05, 1.25)
                else:
                    speed_factor = random.uniform(0.75, 0.95)
            else:
                speed_factor = random.uniform(0.75, 1.25)
        
        speed_factor = max(0.7, min(1.3, speed_factor))
        T = len(sequence)
        new_length = int(T / speed_factor)
        
        if new_length < 5 or new_length == T:
            return sequence
        
        indices = np.linspace(0, T - 1, new_length, dtype=int)
        return sequence[indices].astype(np.float32)
    
    def apply_hand_shift(self, sequence: np.ndarray,
                        max_shift: float = 0.03) -> np.ndarray:
        """Apply spatial shift to hands only"""
        shift_x = random.uniform(-max_shift, max_shift)
        shift_y = random.uniform(-max_shift, max_shift)
        
        result = sequence.astype(np.float32).copy()
        
        if self._has_hand(result[:, self.LH_START:self.LH_END]):
            result[:, self.LH_START:self.LH_END:3] += shift_x
            result[:, self.LH_START+1:self.LH_END:3] += shift_y
        
        if self._has_hand(result[:, self.RH_START:self.RH_END]):
            result[:, self.RH_START:self.RH_END:3] += shift_x
            result[:, self.RH_START+1:self.RH_END:3] += shift_y
        
        return result
    
    def apply_hand_scale(self, sequence: np.ndarray,
                        max_scale_change: float = 0.06) -> np.ndarray:
        """Apply scaling to hands only"""
        scale_x = 1.0 + random.uniform(-max_scale_change, max_scale_change)
        scale_y = 1.0 + random.uniform(-max_scale_change, max_scale_change)
        
        result = sequence.astype(np.float32).copy()
        
        lh = result[:, self.LH_START:self.LH_END]
        rh = result[:, self.RH_START:self.RH_END]
        
        if self._has_hand(lh):
            lh_reshaped = lh.reshape(lh.shape[0], -1, 3)
            mean_x = lh_reshaped[:, :, 0].mean()
            mean_y = lh_reshaped[:, :, 1].mean()
            result[:, self.LH_START:self.LH_END:3] = mean_x + (lh_reshaped[:, :, 0] - mean_x) * scale_x
            result[:, self.LH_START+1:self.LH_END:3] = mean_y + (lh_reshaped[:, :, 1] - mean_y) * scale_y
        
        if self._has_hand(rh):
            rh_reshaped = rh.reshape(rh.shape[0], -1, 3)
            mean_x = rh_reshaped[:, :, 0].mean()
            mean_y = rh_reshaped[:, :, 1].mean()
            result[:, self.RH_START:self.RH_END:3] = mean_x + (rh_reshaped[:, :, 0] - mean_x) * scale_x
            result[:, self.RH_START+1:self.RH_END:3] = mean_y + (rh_reshaped[:, :, 1] - mean_y) * scale_y
        
        return result
    
    def apply_hand_rotation(self, sequence: np.ndarray,
                           max_angle: float = 10.0) -> np.ndarray:
        """Apply rotation to hands only"""
        angle_deg = random.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        result = sequence.astype(np.float32).copy()
        
        lh = result[:, self.LH_START:self.LH_END]
        rh = result[:, self.RH_START:self.RH_END]
        
        if self._has_hand(lh):
            lh_reshaped = lh.reshape(lh.shape[0], -1, 3)
            mean_x = lh_reshaped[:, :, 0].mean()
            mean_y = lh_reshaped[:, :, 1].mean()
            rel_x = lh_reshaped[:, :, 0] - mean_x
            rel_y = lh_reshaped[:, :, 1] - mean_y
            result[:, self.LH_START:self.LH_END:3] = mean_x + rel_x * cos_a - rel_y * sin_a
            result[:, self.LH_START+1:self.LH_END:3] = mean_y + rel_x * sin_a + rel_y * cos_a
        
        if self._has_hand(rh):
            rh_reshaped = rh.reshape(rh.shape[0], -1, 3)
            mean_x = rh_reshaped[:, :, 0].mean()
            mean_y = rh_reshaped[:, :, 1].mean()
            rel_x = rh_reshaped[:, :, 0] - mean_x
            rel_y = rh_reshaped[:, :, 1] - mean_y
            result[:, self.RH_START:self.RH_END:3] = mean_x + rel_x * cos_a - rel_y * sin_a
            result[:, self.RH_START+1:self.RH_END:3] = mean_y + rel_x * sin_a + rel_y * cos_a
        
        return result
    
    def augment_single(self, sequence: np.ndarray, 
                      augmentation_types: List[str] = None) -> np.ndarray:
        """Augment a single sequence"""
        if augmentation_types is None:
            augmentation_types = ['shift', 'scale']
        
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence, dtype=np.float32)
        sequence = sequence.astype(np.float32)
        
        result_seq = sequence.copy()
        
        for aug_type in augmentation_types:
            if aug_type == 'speed':
                result_seq = self.apply_speed_variance(result_seq)
            elif aug_type == 'shift':
                result_seq = self.apply_hand_shift(result_seq)
            elif aug_type == 'scale':
                result_seq = self.apply_hand_scale(result_seq)
            elif aug_type == 'rotation':
                result_seq = self.apply_hand_rotation(result_seq)
        
        if self._is_valid(result_seq):
            self.stats['augmented'] += 1
            return result_seq.astype(np.float32)
        else:
            self.stats['failed'] += 1
            return sequence


# ============================================================
# DATA LOADER FOR NPZ FILES
# ============================================================

def load_npz_data(directory: str) -> Tuple[List, List]:
    """Load data and labels from .npz files in a directory"""
    data_path = Path(directory)
    files = sorted(data_path.glob("*.npz"))
    
    if not files:
        raise ValueError(f"No .npz files found in {directory}")
    
    print(f"📁 Loading {len(files)} files from {directory}...")
    
    all_data = []
    all_labels = []
    
    for file in tqdm(files, desc="Loading files"):
        try:
            with np.load(file) as npz_file:
                if 'x' in npz_file:
                    data = npz_file['x'].astype(np.float32)
                elif 'data' in npz_file:
                    data = npz_file['data'].astype(np.float32)
                else:
                    keys = list(npz_file.keys())
                    data = npz_file[keys[0]].astype(np.float32)
                
                if 'y' in npz_file:
                    label = int(npz_file['y'])
                elif 'label' in npz_file:
                    label = npz_file['label']
                else:
                    label = file.stem
                
                all_data.append(data)
                all_labels.append(label)
        except Exception as e:
            print(f"⚠️ Error loading {file.name}: {e}")
    
    print(f"✅ Loaded {len(all_data)} samples")
    return all_data, all_labels


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def apply_augmentation_to_list(data, labels, 
                               augmentations_per_sample=2, 
                               augmentation_types=['shift', 'scale'], 
                               preserve_original=True,
                               verbose=True):
    """Apply augmentation to list of arrays"""
    augmenter = MemoryEfficientAugmenter()
    
    augmented_data = []
    augmented_labels = []
    num_samples = len(data)
    
    if verbose:
        print(f"Processing {num_samples} samples...")
    
    for i in range(num_samples):
        sample = data[i]
        label = labels[i]
        
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample, dtype=np.float32)
        else:
            sample = sample.astype(np.float32)
        
        if preserve_original:
            augmented_data.append(sample)
            augmented_labels.append(label)
        
        for aug_idx in range(augmentations_per_sample):
            if len(augmentation_types) > 2:
                selected = random.sample(augmentation_types, random.randint(1, 2))
            else:
                selected = augmentation_types
            
            aug_sample = augmenter.augment_single(sample, selected)
            augmented_data.append(aug_sample)
            augmented_labels.append(label)
        
        if verbose and (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{num_samples} samples")
        
        if (i + 1) % 1000 == 0:
            gc.collect()
    
    if verbose:
        print(f"\n📊 Augmentation Stats:")
        print(f"   Original samples: {num_samples}")
        print(f"   Augmented samples: {len(augmented_data)}")
        print(f"   Augmentation factor: {len(augmented_data)/num_samples:.1f}x")
    
    return augmented_data, np.array(augmented_labels)


def save_augmented_data(X_augmented, y_augmented, output_dir: str, verbose=True):
    """Save augmented data in loader-compatible format"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create label encoding
    unique_labels = sorted(set(y_augmented))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    if verbose:
        print(f"\n📊 Label encoding:")
        print(f"   Unique classes: {len(unique_labels)}")
    
    # Save each sample
    for idx, (data, label_str) in enumerate(tqdm(zip(X_augmented, y_augmented), 
                                                  total=len(X_augmented),
                                                  desc="Saving files")):
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        
        label_int = label_to_idx[label_str]
        
        np.savez_compressed(
            output_path / f"{idx}.npz",
            x=data.astype(np.float32),
            y=label_int
        )
    
    # Save label encoder
    np.save(output_path / "label_encoder.npy", label_to_idx)
    
    if verbose:
        print(f"\n✅ Saved {len(X_augmented)} files to {output_path}/")
        print(f"✅ Saved label encoder to {output_path}/label_encoder.npy")
    
    return label_to_idx, idx_to_label


def verify_saved_data(output_dir: str, label_to_idx: dict, verbose=True):
    """Verify saved files match expected format"""
    output_path = Path(output_dir)
    first_file = output_path / "0.npz"
    
    if not first_file.exists():
        print(f"❌ First file not found: {first_file}")
        return False
    
    with np.load(first_file) as test_load:
        if verbose:
            print(f"\n✅ Verification:")
            print(f"   Keys: {list(test_load.keys())}")
            print(f"   'x' shape: {test_load['x'].shape}")
            print(f"   'y' value: {test_load['y']}")
        
        expected_keys = {'x', 'y'}
        actual_keys = set(test_load.keys())
        
        if expected_keys.issubset(actual_keys):
            if verbose:
                print(f"   ✅ Keys match! ('x' and 'y' present)")
            return True
        else:
            print(f"   ❌ Keys don't match! Expected {expected_keys}, got {actual_keys}")
            return False


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Augment sign language landmark data')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to input directory containing .npz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory for augmented data')
    parser.add_argument('--augmentations-per-sample', type=int, default=3,
                        help='Number of augmentations per sample (default: 3)')
    parser.add_argument('--augmentation-types', type=str, nargs='+', 
                        default=['shift', 'scale', 'rotation'],
                        help='Augmentation types: shift, scale, rotation, speed')
    parser.add_argument('--preserve-original', action='store_true', default=True,
                        help='Keep original samples (default: True)')
    parser.add_argument('--no-preserve-original', dest='preserve_original', 
                        action='store_false', help='Don\'t keep original samples')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUGMENTATION PIPELINE FOR SIGN LANGUAGE DATA")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentations per sample: {args.augmentations_per_sample}")
    print(f"Augmentation types: {args.augmentation_types}")
    print(f"Preserve original: {args.preserve_original}")
    print("="*60)
    
    # Step 1: Load data
    print("\n STEP 1: Loading data...")
    all_data, all_labels = load_npz_data(args.input_dir)
    
    print(f"\n Dataset stats:")
    print(f"   Total samples: {len(all_data)}")
    print(f"   Unique labels: {len(set(all_labels))}")
    print(f"   First sample shape: {all_data[0].shape}")
    
    # Step 2: Apply augmentation
    print("\nSTEP 2: Applying augmentation...")
    gc.collect()
    
    X_augmented, y_augmented = apply_augmentation_to_list(
        data=all_data,
        labels=all_labels,
        augmentations_per_sample=args.augmentations_per_sample,
        augmentation_types=args.augmentation_types,
        preserve_original=args.preserve_original,
        verbose=True
    )
    
    print(f"\n Augmentation complete!")
    print(f"   Original samples: {len(all_data)}")
    print(f"   Augmented samples: {len(X_augmented)}")
    print(f"   Augmentation factor: {len(X_augmented)/len(all_data):.1f}x")
    
    # Step 3: Save augmented data
    print("\n STEP 3: Saving augmented data...")
    label_to_idx, idx_to_label = save_augmented_data(
        X_augmented, y_augmented, args.output_dir, verbose=True
    )
    
    # Step 4: Verify
    print("\n STEP 4: Verifying saved data...")
    verify_saved_data(args.output_dir, label_to_idx, verbose=True)
    
    # Step 5: Summary
    print("\n" + "="*60)
    print(" AUGMENTATION COMPLETE!")
    print("="*60)
    print(f"Output directory: {args.output_dir}/")
    print(f"Total files saved: {len(X_augmented)}")
    print(f"File naming: 0.npz, 1.npz, ...")
    print(f"Each file contains: 'x' (data), 'y' (integer label)")
    print(f"Label encoder: {args.output_dir}/label_encoder.npy")
    print("\nReady to use with training script!")
    print("="*60)


if __name__ == "__main__":
    main()