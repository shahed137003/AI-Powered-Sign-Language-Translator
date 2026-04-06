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
# EFFICIENT DATA LOADER CLASS
# ============================================================================

class SignLanguageDataLoader:
    """
    Efficient data loader for sign language landmark data
    
    Features:
    - Lazy loading (load only when needed)
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
        self.labels = []
        self.label_to_indices = defaultdict(list)
        self.metadata = None
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load file metadata without loading actual data"""
        cache_file = self.cache_path / 'metadata.pkl'
        
        # Try to load from cache
        if cache_file.exists():
            print(f"📦 Loading metadata from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                metadata = pickle.load(f)
                self.file_list = metadata['file_list']
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
        
        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Loading metadata"):
            # Extract label
            name = file_path.stem
            label = re.sub(r'\s+\d+$', '', name)
            
            self.file_list.append(file_path)
            self.labels.append(label)
            self.label_to_indices[label].append(len(self.file_list) - 1)
        
        # Calculate statistics
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.metadata = {
            'total_files': len(self.file_list),
            'unique_labels': len(unique_labels),
            'label_counts': dict(zip(unique_labels, counts)),
            'avg_samples_per_label': len(self.file_list) / len(unique_labels),
            'min_samples': counts.min(),
            'max_samples': counts.max()
        }
        
        # Cache metadata
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'file_list': self.file_list,
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
                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for specified indices
        
        Parameters:
        - indices: List of indices to load (None = load all)
        - verbose: Print progress
        
        Returns:
        - Tuple of (data_array, labels_array)
        """
        if indices is None:
            indices = list(range(len(self.file_list)))
        
        data_list = []
        labels_list = []
        
        iterator = tqdm(indices, desc="Loading data") if verbose else indices
        
        for idx in iterator:
            file_path = self.file_list[idx]
            data = np.load(file_path)
            data_list.append(data)
            labels_list.append(self.labels[idx])
        
        return np.array(data_list, dtype=object), np.array(labels_list)
    
    def load_batch(self, batch_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific batch of data
        
        Parameters:
        - batch_indices: List of indices to load
        
        Returns:
        - Tuple of (data_array, labels_array)
        """
        return self.load_data(batch_indices, verbose=False)
    
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
        
        data, labels = self.load_data(indices, verbose=True)
        
        # Save as numpy files
        np.save(output_path / 'data.npy', data, allow_pickle=True)
        np.save(output_path / 'labels.npy', labels)
        
        # Save metadata
        metadata = {
            'total_samples': len(data),
            'unique_labels': len(np.unique(labels)),
            'label_counts': dict(zip(*np.unique(labels, return_counts=True)))
        }
        
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Data saved to {output_path}")
        print(f"   Data shape: {data.shape}")
        print(f"   Labels shape: {labels.shape}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_data_efficient(dataset_path: str, 
                       min_frequency: int = 1,
                       cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-line function to load data efficiently
    
    Parameters:
    - dataset_path: Path to dataset
    - min_frequency: Minimum frequency per label (filter rare labels)
    - cache: Use caching for faster subsequent loads
    
    Returns:
    - Tuple of (data, labels)
    """
    loader = SignLanguageDataLoader(dataset_path)
    
    if min_frequency > 1:
        indices = loader.filter_by_frequency(min_count=min_frequency)
        print(f"Filtered to {len(indices)} samples (min frequency: {min_frequency})")
    else:
        indices = None
    
    return loader.load_data(indices)

def quick_load(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick load without any filtering
    """
    return load_data_efficient(dataset_path, min_frequency=1)