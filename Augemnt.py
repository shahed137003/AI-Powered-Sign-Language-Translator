# ============================================================
# MEMORY-EFFICIENT LANDMARK AUGMENTATION (NO MASKS)
# ============================================================

import numpy as np
import random
from typing import Tuple, List, Optional, Union
import warnings
import gc
from tqdm import tqdm
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
            # Convert object array to float32
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
# MEMORY-EFFICIENT AUGMENTER (NO MASKS)
# ============================================================

class MemoryEfficientAugmenter:
    """
    Memory-efficient augmentation for sign language landmarks (no masks)
    - Processes one sample at a time
    - Only modifies hand landmarks (preserves face and pose)
    - Validates all augmentations
    """
    
    def __init__(self, feature_dim: int = FEATURE_DIM):
        self.feature_dim = feature_dim
        
        # Feature indices
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
        
        # Statistics
        self.stats = {
            'augmented': 0,
            'failed': 0
        }
    
    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    
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
    
    # ============================================================
    # SPEED VARIANCE (Time Warping)
    # ============================================================
    
    def apply_speed_variance(self, sequence: np.ndarray, 
                             speed_factor: float = None,
                             mode: str = 'balanced') -> np.ndarray:
        """
        Apply speed variance using frame sampling (no interpolation artifacts)
        
        Args:
            sequence: Input sequence
            speed_factor: Custom speed factor
            mode: 'balanced', 'faster', 'slower', 'random'
        
        Returns:
            augmented_sequence
        """
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
        
        # Sample sequence at indices
        augmented_sequence = sequence[indices].astype(np.float32)
        
        return augmented_sequence
    
    # ============================================================
    # HAND SHIFTING (Spatial Translation)
    # ============================================================
    
    def apply_hand_shift(self, sequence: np.ndarray,
                        max_shift: float = 0.03) -> np.ndarray:
        """
        Apply spatial shift to hands only
        """
        shift_x = random.uniform(-max_shift, max_shift)
        shift_y = random.uniform(-max_shift, max_shift)
        
        result = sequence.astype(np.float32).copy()
        
        # Apply shift to hand landmarks only
        if self._has_hand(result[:, self.LH_START:self.LH_END]):
            result[:, self.LH_START:self.LH_END:3] += shift_x
            result[:, self.LH_START+1:self.LH_END:3] += shift_y
        
        if self._has_hand(result[:, self.RH_START:self.RH_END]):
            result[:, self.RH_START:self.RH_END:3] += shift_x
            result[:, self.RH_START+1:self.RH_END:3] += shift_y
        
        return result
    
    # ============================================================
    # HAND SCALING
    # ============================================================
    
    def apply_hand_scale(self, sequence: np.ndarray,
                        max_scale_change: float = 0.06) -> np.ndarray:
        """
        Apply scaling to hands only
        """
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
    
    # ============================================================
    # HAND ROTATION
    # ============================================================
    
    def apply_hand_rotation(self, sequence: np.ndarray,
                           max_angle: float = 10.0) -> np.ndarray:
        """
        Apply rotation to hands only
        """
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
    
    # ============================================================
    # SINGLE AUGMENTATION
    # ============================================================
    
    def augment_single(self, sequence: np.ndarray, 
                      augmentation_types: List[str] = None) -> np.ndarray:
        """
        Augment a single sequence (no mask)
        
        Returns:
            augmented_sequence
        """
        if augmentation_types is None:
            augmentation_types = ['shift', 'scale']
        
        # Ensure data is proper type
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
# BATCH AUGMENTATION (DATA ONLY)
# ============================================================

def augment_dataset(data: np.ndarray, 
                    labels: np.ndarray = None,
                    augmentations_per_sample: int = 2,
                    augmentation_types: List[str] = None,
                    batch_size: int = 100,
                    preserve_original: bool = True,
                    verbose: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Memory-efficient batch augmentation (data only, no masks)
    
    Args:
        data: Input data array (num_samples, frames, features)
        labels: Optional labels
        augmentations_per_sample: Number of augmentations per sample
        augmentation_types: List of augmentation types to apply
        batch_size: Batch size for processing
        preserve_original: Whether to keep original samples
        verbose: Print progress
    
    Returns:
        If labels provided: (augmented_data, augmented_labels)
        Else: augmented_data
    """
    # Validate and convert data
    print("🔧 Validating data format...")
    data = validate_and_convert_data(data)
    
    augmenter = MemoryEfficientAugmenter()
    
    if augmentation_types is None:
        augmentation_types = ['shift', 'scale']
    
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_augmented = []
    all_labels = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        
        if verbose:
            print(f"Processing batch {batch_idx+1}/{num_batches} (samples {start}-{end})")
        
        batch_data = data[start:end]
        batch_labels = labels[start:end] if labels is not None else None
        
        for i, sample in enumerate(batch_data):
            # Add original
            if preserve_original:
                all_augmented.append(sample.astype(np.float32))
                if batch_labels is not None:
                    all_labels.append(batch_labels[i])
            
            # Create augmentations
            for aug_idx in range(augmentations_per_sample):
                # Randomly select 1-2 augmentation types
                if len(augmentation_types) > 2:
                    selected = random.sample(augmentation_types, random.randint(1, 2))
                else:
                    selected = augmentation_types
                
                aug_sample = augmenter.augment_single(sample, selected)
                
                all_augmented.append(aug_sample.astype(np.float32))
                if batch_labels is not None:
                    all_labels.append(batch_labels[i])
        
        # Clear memory
        gc.collect()
    
    # Convert to numpy arrays
    result_data = np.array(all_augmented, dtype=np.float32)
    result_labels = np.array(all_labels) if labels is not None else None
    
    if verbose:
        print(f"\n📊 Augmentation Stats:")
        print(f"   Original samples: {num_samples}")
        print(f"   Augmented samples: {len(result_data)}")
        print(f"   Augmentation factor: {len(result_data)/num_samples:.1f}x")
        print(f"   Success rate: {augmenter.stats['augmented']/max(1, augmenter.stats['augmented']+augmenter.stats['failed'])*100:.1f}%")
    
    if result_labels is not None:
        return result_data, result_labels
    return result_data


# ============================================================
# SIMPLE AUGMENTATION (MOST MEMORY EFFICIENT - NO MASKS)
# ============================================================

def simple_augment(data: np.ndarray, 
                   labels: np.ndarray = None,
                   augmentations_per_sample: int = 2,
                   augmentation_types: List[str] = None,
                   verbose: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Simple augmentation - processes one sample at a time (most memory efficient)
    No masks
    
    Args:
        data: Input data array
        labels: Optional labels
        augmentations_per_sample: Number of augmentations per sample
        augmentation_types: List of augmentation types
        verbose: Print progress
    
    Returns:
        If labels provided: (augmented_data, augmented_labels)
        Else: augmented_data
    """
    # Validate data
    data = validate_and_convert_data(data)
    
    augmenter = MemoryEfficientAugmenter()
    
    if augmentation_types is None:
        augmentation_types = ['shift', 'scale']
    
    augmented_data = []
    augmented_labels = []
    
    iterator = tqdm(range(len(data)), desc="Augmenting") if verbose else range(len(data))
    
    for i in iterator:
        sample = data[i]
        sample_label = labels[i] if labels is not None else None
        
        # Add original
        augmented_data.append(sample)
        if sample_label is not None:
            augmented_labels.append(sample_label)
        
        # Create augmentations
        for aug_idx in range(augmentations_per_sample):
            if len(augmentation_types) > 2:
                selected = random.sample(augmentation_types, random.randint(1, 2))
            else:
                selected = augmentation_types
            
            aug_sample = augmenter.augment_single(sample, selected)
            
            augmented_data.append(aug_sample)
            if sample_label is not None:
                augmented_labels.append(sample_label)
        
        # Clear memory periodically
        if (i + 1) % 1000 == 0:
            gc.collect()
    
    result_data = np.array(augmented_data, dtype=np.float32)
    result_labels = np.array(augmented_labels) if labels is not None else None
    
    if verbose:
        print(f"\n📊 Augmentation Stats:")
        print(f"   Original: {len(data)} → Augmented: {len(result_data)}")
        print(f"   Factor: {len(result_data)/len(data):.1f}x")
    
    if result_labels is not None:
        return result_data, result_labels
    return result_data


