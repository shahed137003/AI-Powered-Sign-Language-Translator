import os
import numpy as np
import re
from collections import defaultdict
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EFFICIENT DATA LOADER CLASS (WITH MASKS)
# ============================================================================

class SignLanguageDataLoader:
    """
    Efficient data loader for sign language landmark data with masks
    
    Features:
    - Lazy loading (load only when needed)
    - Automatic mask loading
    - Caching support
    - Train/val/test split
    - Filter by label frequency
    - Batch processing
    """
    
    def __init__(self, dataset_path: str, cache_path: Optional[str] = None):
        """
        Initialize the data loader
        
        Parameters:
        - dataset_path: Path to the directory containing .npy files
        - cache_path: Optional path to save/load cached metadata
        """
        self.dataset_path = Path(dataset_path)
        self.cache_path = Path(cache_path) if cache_path else self.dataset_path / '.cache'
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        self.file_list = []
        self.mask_list = []  # Store mask file paths
        self.labels = []
        self.label_to_indices = defaultdict(list)
        self.metadata = None
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load file metadata without loading actual data"""
        cache_file = self.cache_path / 'metadata_with_masks.pkl'
        
        # Try to load from cache
        if cache_file.exists():
            print(f"📦 Loading metadata from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                metadata = pickle.load(f)
                self.file_list = metadata['file_list']
                self.mask_list = metadata['mask_list']
                self.labels = metadata['labels']
                self.label_to_indices = metadata['label_to_indices']
                self.metadata = metadata['stats']
            print(f"✅ Loaded metadata for {len(self.file_list)} files")
            return
        
        # Scan directory
        print(f"🔍 Scanning directory: {self.dataset_path}")
        all_files = list(self.dataset_path.glob("*.npy"))
        
        if not all_files:
            raise ValueError(f"No .npy files found in {self.dataset_path}")
        
        print(f"📁 Found {len(all_files)} .npy files")
        
        # Separate data files and mask files
        data_files = []
        mask_files_dict = {}
        
        for file_path in all_files:
            filename = file_path.stem
            if filename.endswith('_mask'):
                # This is a mask file
                original_name = filename[:-5]  # Remove '_mask' suffix
                mask_files_dict[original_name] = file_path
            else:
                # This is a data file
                data_files.append(file_path)
        
        print(f"   Data files: {len(data_files)}")
        print(f"   Mask files: {len(mask_files_dict)}")
        
        # Process data files with progress bar
        for file_path in tqdm(data_files, desc="Loading metadata"):
            # Extract label
            name = file_path.stem
            label = re.sub(r'\s+\d+$', '', name)
            label = re.sub(r'_\d+$', '', label)
            
            self.file_list.append(file_path)
            
            # Find corresponding mask
            original_name = file_path.stem
            if original_name in mask_files_dict:
                self.mask_list.append(mask_files_dict[original_name])
            else:
                # No mask found, will create default mask later
                self.mask_list.append(None)
            
            self.labels.append(label)
            self.label_to_indices[label].append(len(self.file_list) - 1)
        
        # Calculate statistics
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        valid_masks = sum(1 for m in self.mask_list if m is not None)
        
        self.metadata = {
            'total_files': len(self.file_list),
            'unique_labels': len(unique_labels),
            'label_counts': dict(zip(unique_labels, counts)),
            'avg_samples_per_label': len(self.file_list) / len(unique_labels),
            'min_samples': counts.min(),
            'max_samples': counts.max(),
            'masks_available': valid_masks,
            'masks_missing': len(self.file_list) - valid_masks
        }
        
        # Cache metadata
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'file_list': self.file_list,
                'mask_list': self.mask_list,
                'labels': self.labels,
                'label_to_indices': self.label_to_indices,
                'stats': self.metadata
            }, f)
        
        print(f"✅ Metadata cached to {cache_file}")
    
    def get_label_stats(self) -> pd.DataFrame:
        """Get label distribution statistics"""
        label_counts = pd.DataFrame([
            {'label': label, 'count': len(indices)}
            for label, indices in self.label_to_indices.items()
        ]).sort_values('count', ascending=False)
        
        return label_counts
    
    def filter_by_frequency(self, min_count: int = 5, max_count: int = None) -> List[int]:
        """
        Filter indices by label frequency
        
        Parameters:
        - min_count: Minimum number of samples per label
        - max_count: Maximum number of samples per label (optional)
        
        Returns:
        - List of indices that meet the criteria
        """
        valid_indices = []
        
        for label, indices in self.label_to_indices.items():
            if len(indices) >= min_count:
                if max_count is None or len(indices) <= max_count:
                    valid_indices.extend(indices)
        
        return sorted(valid_indices)
    
    def load_data(self, indices: Optional[List[int]] = None, 
                  verbose: bool = True,
                  load_masks: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and masks for specified indices
        
        Parameters:
        - indices: List of indices to load (None = load all)
        - verbose: Print progress
        - load_masks: Whether to load mask files
        
        Returns:
        - Tuple of (data_array, labels_array, masks_array)
        """
        if indices is None:
            indices = list(range(len(self.file_list)))
        
        data_list = []
        labels_list = []
        masks_list = []
        
        iterator = tqdm(indices, desc="Loading data") if verbose else indices
        
        for idx in iterator:
            file_path = self.file_list[idx]
            data = np.load(file_path)
            data_list.append(data)
            labels_list.append(self.labels[idx])
            
            # Load mask if available
            if load_masks:
                mask_path = self.mask_list[idx]
                if mask_path is not None and mask_path.exists():
                    mask = np.load(mask_path)
                    # Ensure mask is 1D
                    if mask.ndim > 1:
                        mask = mask.flatten()
                    masks_list.append(mask)
                else:
                    # Create default mask (all ones)
                    mask = np.ones(data.shape[0], dtype=np.float32)
                    masks_list.append(mask)
        
        data_array = np.array(data_list, dtype=object)
        labels_array = np.array(labels_list)
        masks_array = np.array(masks_list, dtype=object) if load_masks else None
        
        return data_array, labels_array, masks_array
    
    def load_batch(self, batch_indices: List[int], 
                   load_masks: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a specific batch of data
        
        Parameters:
        - batch_indices: List of indices to load
        - load_masks: Whether to load masks
        
        Returns:
        - Tuple of (data_array, labels_array, masks_array)
        """
        return self.load_data(batch_indices, verbose=False, load_masks=load_masks)
    
    def create_stratified_split(self, train_ratio: float = 0.7, 
                                val_ratio: float = 0.15, 
                                test_ratio: float = 0.15,
                                random_seed: int = 42) -> Dict[str, List[int]]:
        """
        Create stratified train/val/test split based on labels
        
        Parameters:
        - train_ratio: Proportion for training
        - val_ratio: Proportion for validation
        - test_ratio: Proportion for testing
        - random_seed: Random seed for reproducibility
        
        Returns:
        - Dictionary with 'train', 'val', 'test' keys containing indices
        """
        np.random.seed(random_seed)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for label, indices in self.label_to_indices.items():
            n_total = len(indices)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Shuffle indices
            shuffled = np.random.permutation(indices)
            
            train_indices.extend(shuffled[:n_train])
            val_indices.extend(shuffled[n_train:n_train + n_val])
            test_indices.extend(shuffled[n_train + n_val:])
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def get_class_weights(self) -> Dict[str, float]:
        """
        Calculate class weights for imbalanced dataset
        """
        total_samples = len(self.file_list)
        n_classes = len(self.label_to_indices)
        
        class_weights = {}
        for label, indices in self.label_to_indices.items():
            weight = total_samples / (n_classes * len(indices))
            class_weights[label] = weight
        
        return class_weights
    
    def summary(self) -> None:
        """Print summary of the dataset"""
        print("="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        print(f"Path: {self.dataset_path}")
        print(f"Total files: {self.metadata['total_files']}")
        print(f"Unique glosses: {self.metadata['unique_labels']}")
        print(f"Average samples/gloss: {self.metadata['avg_samples_per_label']:.2f}")
        print(f"Min samples/gloss: {self.metadata['min_samples']}")
        print(f"Max samples/gloss: {self.metadata['max_samples']}")
        print(f"Masks available: {self.metadata['masks_available']}")
        print(f"Masks missing: {self.metadata['masks_missing']}")
        
        # Show distribution
        label_counts = self.get_label_stats()
        print(f"\n📈 Label Distribution:")
        print(f"  Top 10 most frequent:")
        for i, row in label_counts.head(10).iterrows():
            print(f"    {row['label'][:35]:<35} {row['count']:>5} samples")
        
        print(f"\n  Bottom 10 least frequent:")
        for i, row in label_counts.tail(10).iterrows():
            print(f"    {row['label'][:35]:<35} {row['count']:>5} samples")
        
        # Show quality metrics
        print(f"\n🔍 Quality Metrics:")
        print(f"  Labels with <5 samples: {(label_counts['count'] < 5).sum()}")
        print(f"  Labels with <10 samples: {(label_counts['count'] < 10).sum()}")
        print(f"  Labels with >100 samples: {(label_counts['count'] > 100).sum()}")
    
    def save_processed_data(self, output_path: str, indices: Optional[List[int]] = None):
        """
        Save processed data to disk in efficient format
        
        Parameters:
        - output_path: Path to save the data
        - indices: Indices to save (None = save all)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data, labels, masks = self.load_data(indices, verbose=True)
        
        # Create subdirectories for splits
        (output_path / 'data').mkdir(exist_ok=True)
        (output_path / 'masks').mkdir(exist_ok=True)
        
        # Save each sample individually
        for i, (d, l, m) in enumerate(zip(data, labels, masks)):
            safe_label = l.replace(' ', '_').replace('/', '_')
            np.save(output_path / 'data' / f"{safe_label}_{i:06d}.npy", d)
            np.save(output_path / 'masks' / f"{safe_label}_{i:06d}_mask.npy", m)
        
        # Save metadata
        metadata = {
            'total_samples': len(data),
            'unique_labels': len(np.unique(labels)),
            'label_counts': dict(zip(*np.unique(labels, return_counts=True))),
            'data_shape': data[0].shape if len(data) > 0 else None
        }
        
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Data saved to {output_path}")
        print(f"   Total samples: {len(data)}")
        print(f"   Unique labels: {metadata['unique_labels']}")

# ============================================================================
# CONVENIENCE FUNCTIONS (WITH MASKS)
# ============================================================================

def load_data_efficient(dataset_path: str, 
                       min_frequency: int = 1,
                       cache: bool = True,
                       load_masks: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-line function to load data efficiently with masks
    
    Parameters:
    - dataset_path: Path to dataset
    - min_frequency: Minimum frequency per label (filter rare labels)
    - cache: Use caching for faster subsequent loads
    - load_masks: Whether to load mask files
    
    Returns:
    - Tuple of (data, labels, masks)
    """
    loader = SignLanguageDataLoader(dataset_path, cache_path=dataset_path + '/.cache' if cache else None)
    
    if min_frequency > 1:
        indices = loader.filter_by_frequency(min_count=min_frequency)
        print(f"Filtered to {len(indices)} samples (min frequency: {min_frequency})")
    else:
        indices = None
    
    return loader.load_data(indices, load_masks=load_masks)

def quick_load(dataset_path: str, load_masks: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick load without any filtering
    
    Parameters:
    - dataset_path: Path to dataset
    - load_masks: Whether to load masks
    
    Returns:
    - Tuple of (data, labels, masks)
    """
    return load_data_efficient(dataset_path, min_frequency=1, load_masks=load_masks)

def load_data_and_masks(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load both data and masks
    
    Returns:
    - Tuple of (data, labels, masks)
    """
    return load_data_efficient(dataset_path, load_masks=True)

def load_data_only(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load only data and labels (no masks)
    
    Returns:
    - Tuple of (data, labels)
    """
    data, labels, _ = load_data_efficient(dataset_path, load_masks=False)
    return data, labels

