# ============================================================================
# MEMORY-EFFICIENT DATA LOADER FOR NPZ FILES (DATA + LABELS ONLY)
# ============================================================================

import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import Tuple, List, Optional, Generator
import warnings
warnings.filterwarnings('ignore')

class SignLanguageDataLoader:
    """
    Memory-efficient data loader for sign language landmark data (.npz files)
    Handles ONLY data and labels (no masks)
    """
    
    def __init__(self, dataset_path: str, cache_path: Optional[str] = None, batch_size: int = 500):
        self.dataset_path = Path(dataset_path)
        self.cache_path = Path(cache_path) if cache_path else self.dataset_path / '.cache'
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        self.file_list = []
        self.labels = []
        self.label_to_indices = defaultdict(list)
        self.metadata = None
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load only metadata (file paths), not the actual data"""
        cache_file = self.cache_path / 'metadata.pkl'
        
        if cache_file.exists():
            print(f"📦 Loading metadata from cache...")
            with open(cache_file, 'rb') as f:
                metadata = pickle.load(f)
                self.file_list = metadata['file_list']
                self.labels = metadata['labels']
                self.label_to_indices = metadata['label_to_indices']
                self.metadata = metadata['stats']
            print(f"✅ Loaded metadata for {len(self.file_list)} files")
            return
        
        print(f"🔍 Scanning directory: {self.dataset_path}")
        all_files = list(self.dataset_path.glob("*.npz"))
        
        if not all_files:
            raise ValueError(f"No .npz files found in {self.dataset_path}")
        
        print(f"📁 Found {len(all_files)} .npz files")
        
        # Process metadata - load label from each file
        for file_path in tqdm(all_files, desc="Loading metadata"):
            try:
                with np.load(file_path) as npz_file:
                    # Extract label (should be 'y' key)
                    if 'y' in npz_file:
                        label = int(npz_file['y'])  # Convert to Python int
                    else:
                        # Fallback to filename if no label found
                        label = file_path.stem
                    
                    self.file_list.append(file_path)
                    self.labels.append(label)
                    self.label_to_indices[str(label)].append(len(self.file_list) - 1)
                    
            except Exception as e:
                print(f"⚠️ Error reading {file_path.name}: {e}")
                continue
        
        # Statistics
        unique_labels = list(set(self.labels))
        counts = [self.labels.count(l) for l in unique_labels]
        
        self.metadata = {
            'total_files': len(self.file_list),
            'unique_labels': len(unique_labels),
            'label_counts': dict(zip(unique_labels, counts)),
            'avg_samples_per_label': len(self.file_list) / len(unique_labels) if unique_labels else 0,
            'min_samples': min(counts) if counts else 0,
            'max_samples': max(counts) if counts else 0,
        }
        
        # Cache metadata
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'file_list': self.file_list,
                'labels': self.labels,
                'label_to_indices': self.label_to_indices,
                'stats': self.metadata
            }, f)
        
        print(f"✅ Metadata cached")
    
    def load_single(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Load a single sample from .npz file (memory efficient)
        
        Returns:
            data: numpy array of shape (frames, features)
            label: integer label (already encoded)
        """
        # Load npz file
        with np.load(self.file_list[idx]) as npz_file:
            # Get data - should be 'x' key
            if 'x' in npz_file:
                data = npz_file['x'].astype(np.float32)
            elif 'data' in npz_file:
                data = npz_file['data'].astype(np.float32)
            else:
                # If no standard key, try the first available key
                keys = list(npz_file.keys())
                if keys:
                    data = npz_file[keys[0]].astype(np.float32)
                else:
                    raise ValueError(f"No data array found in {self.file_list[idx]}")
            
            # Get label - should be 'y' key (integer)
            if 'y' in npz_file:
                label = int(npz_file['y'])  # Convert to Python int
            else:
                # Fallback to stored label
                label = self.labels[idx]
        
        return data, label
    
    def load_all(self, verbose: bool = True) -> Tuple[List, List]:
        """
        ⚠️ WARNING: This loads ALL data into memory (may crash!)
        Use iterate_batches() instead for large datasets
        """
        print("⚠️ WARNING: Loading all data into memory. This may crash if dataset is large!")
        
        all_data = []
        all_labels = []
        
        iterator = tqdm(range(len(self.file_list)), desc="Loading all data") if verbose else range(len(self.file_list))
        
        for idx in iterator:
            data, label = self.load_single(idx)
            all_data.append(data)
            all_labels.append(label)
        
        return all_data, all_labels
    
    def iterate_batches(self, batch_size: int = None) -> Generator:
        """
        Generator that yields batches of data - MEMORY EFFICIENT!
        
        Yields:
            batch_data: List of data arrays
            batch_labels: List of integer labels
        """
        batch_size = batch_size or self.batch_size
        
        for start_idx in range(0, len(self.file_list), batch_size):
            end_idx = min(start_idx + batch_size, len(self.file_list))
            batch_data = []
            batch_labels = []
            
            for idx in range(start_idx, end_idx):
                data, label = self.load_single(idx)
                batch_data.append(data)
                batch_labels.append(label)
            
            yield batch_data, batch_labels
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
    
    def inspect_npz_file(self, idx: int = 0):
        """Inspect the structure of an npz file to understand its contents"""
        file_path = self.file_list[idx]
        print(f"\n🔍 Inspecting: {file_path.name}")
        with np.load(file_path) as npz_file:
            print(f"   Keys in file: {list(npz_file.keys())}")
            for key in npz_file.keys():
                arr = npz_file[key]
                print(f"   {key}: shape={arr.shape}, dtype={arr.dtype}")
                if key == 'y':
                    print(f"      Label value: {arr}")
        return None
    
    def summary(self):
        """Print dataset summary"""
        print("="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        print(f"Path: {self.dataset_path}")
        print(f"Total files: {self.metadata['total_files']}")
        print(f"Unique labels: {self.metadata['unique_labels']}")
        print(f"Avg samples/label: {self.metadata['avg_samples_per_label']:.2f}")
        print(f"Min samples: {self.metadata['min_samples']}")
        print(f"Max samples: {self.metadata['max_samples']}")
        print(f"Batch size: {self.batch_size}")
    
    def get_label_distribution(self):
        """Get class distribution as a dictionary"""
        return self.metadata['label_counts']


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_data_efficient(dataset_path: str, 
                       cache: bool = True,
                       batch_size: int = 500):
    """Returns a loader object for .npz files"""
    cache_path = dataset_path + '/.cache' if cache else None
    loader = SignLanguageDataLoader(dataset_path, cache_path=cache_path, batch_size=batch_size)
    return loader


def quick_load(dataset_path: str, batch_size: int = 500):
    """Quick load for .npz files"""
    return load_data_efficient(dataset_path, batch_size=batch_size)


def load_data_and_labels(dataset_path: str, batch_size: int = 500):
    """Returns loader object for .npz files"""
    return SignLanguageDataLoader(dataset_path, batch_size=batch_size)


def inspect_dataset(dataset_path: str, sample_idx: int = 0):
    """Inspect the first few npz files to understand structure"""
    loader = SignLanguageDataLoader(dataset_path)
    loader.inspect_npz_file(sample_idx)
    return loader

