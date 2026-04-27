# ============================================================
# CLASS BALANCING FOR AUGMENTED DATA
# ============================================================

import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. ANALYZE CURRENT CLASS DISTRIBUTION
# ============================================================

def analyze_class_balance(labels):
    """
    Analyze current class distribution
    """
    class_counts = Counter(labels)
    
    print("="*60)
    print("📊 CURRENT CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Total classes: {len(class_counts)}")
    print(f"Total samples: {sum(class_counts.values()):,}")
    print(f"Min samples: {min(class_counts.values())}")
    print(f"Max samples: {max(class_counts.values())}")
    print(f"Mean samples: {np.mean(list(class_counts.values())):.1f}")
    print(f"Std samples: {np.std(list(class_counts.values())):.1f}")
    print(f"Median samples: {np.median(list(class_counts.values())):.1f}")
    
    # Calculate imbalance ratio
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.1f}")
    
    if imbalance_ratio > 3:
        print("⚠️ Significant imbalance detected!")
    else:
        print("✅ Classes are reasonably balanced")
    
    return class_counts

# ============================================================
# 2. OPTION 1: WEIGHTED SAMPLER (During Training)
# ============================================================

def create_weighted_sampler(labels):
    """
    Create a weighted sampler for balanced batch sampling
    This doesn't modify data, just sampling weights
    """
    class_counts = Counter(labels)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[label] for label in labels])
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    print("\n" + "="*60)
    print("📊 WEIGHTED SAMPLER CONFIGURATION")
    print("="*60)
    print(f"Min weight: {sample_weights.min():.6f}")
    print(f"Max weight: {sample_weights.max():.6f}")
    print(f"Mean weight: {sample_weights.mean():.6f}")
    print(f"Weight ratio (max/min): {sample_weights.max()/sample_weights.min():.1f}")
    
    return sample_weights

# ============================================================
# 3. OPTION 2: OVERSAMPLING RARE CLASSES (Modify Data)
# ============================================================

def oversample_rare_classes(data, labels, target_samples_per_class=None, augmenter=None):
    """
    Oversample rare classes to reach target samples per class
    """
    class_counts = Counter(labels)
    
    if target_samples_per_class is None:
        # Target = 80% of max class count
        target_samples_per_class = int(max(class_counts.values()) * 0.8)
    
    print("\n" + "="*60)
    print("📊 OVERSAMPLING RARE CLASSES")
    print("="*60)
    print(f"Target samples per class: {target_samples_per_class}")
    
    balanced_data = []
    balanced_labels = []
    
    oversampled_count = 0
    
    for class_label, current_count in class_counts.items():
        # Get all samples of this class
        class_indices = [i for i, lbl in enumerate(labels) if lbl == class_label]
        class_samples = [data[i] for i in class_indices]
        
        # Add all original samples
        balanced_data.extend(class_samples)
        balanced_labels.extend([class_label] * len(class_samples))
        
        # Calculate how many more needed
        needed = target_samples_per_class - current_count
        
        if needed > 0:
            # Oversample with replacement
            oversample_indices = np.random.choice(len(class_samples), needed, replace=True)
            oversampled_samples = [class_samples[i] for i in oversample_indices]
            
            balanced_data.extend(oversampled_samples)
            balanced_labels.extend([class_label] * needed)
            oversampled_count += needed
            
            print(f"   Class {class_label}: {current_count} → {target_samples_per_class} (+{needed})")
    
    balanced_data = np.array(balanced_data)
    balanced_labels = np.array(balanced_labels)
    
    print(f"\n✅ Oversampling complete!")
    print(f"   Original samples: {len(data)}")
    print(f"   Balanced samples: {len(balanced_data)}")
    print(f"   Oversampled added: {oversampled_count}")
    
    return balanced_data, balanced_labels

# ============================================================
# 4. OPTION 3: UNDERSAMPLING FREQUENT CLASSES (Modify Data)
# ============================================================

def undersample_frequent_classes(data, labels, target_samples_per_class=None):
    """
    Undersample frequent classes to match target
    """
    class_counts = Counter(labels)
    
    if target_samples_per_class is None:
        # Target = 120% of min class count
        target_samples_per_class = int(min(class_counts.values()) * 1.2)
    
    print("\n" + "="*60)
    print("📊 UNDERSAMPLING FREQUENT CLASSES")
    print("="*60)
    print(f"Target samples per class: {target_samples_per_class}")
    
    balanced_data = []
    balanced_labels = []
    
    undersampled_count = 0
    
    for class_label, current_count in class_counts.items():
        # Get all samples of this class
        class_indices = [i for i, lbl in enumerate(labels) if lbl == class_label]
        class_samples = [data[i] for i in class_indices]
        
        if current_count > target_samples_per_class:
            # Undersample
            keep_indices = np.random.choice(current_count, target_samples_per_class, replace=False)
            kept_samples = [class_samples[i] for i in keep_indices]
            balanced_data.extend(kept_samples)
            balanced_labels.extend([class_label] * target_samples_per_class)
            undersampled_count += (current_count - target_samples_per_class)
            print(f"   Class {class_label}: {current_count} → {target_samples_per_class} (-{current_count - target_samples_per_class})")
        else:
            # Keep all
            balanced_data.extend(class_samples)
            balanced_labels.extend([class_label] * current_count)
    
    balanced_data = np.array(balanced_data)
    balanced_labels = np.array(balanced_labels)
    
    print(f"\n✅ Undersampling complete!")
    print(f"   Original samples: {len(data)}")
    print(f"   Balanced samples: {len(balanced_data)}")
    print(f"   Undersampled removed: {undersampled_count}")
    
    return balanced_data, balanced_labels

# ============================================================
# 5. OPTION 4: SMART BALANCING (Recommended)
# ============================================================

def smart_class_balancing(data, labels, 
                         target_mean=None,
                         oversample_threshold=0.7,  # Classes below 70% of mean get oversampled
                         undersample_threshold=1.5,  # Classes above 150% of mean get undersampled
                         use_augmentation=True,
                         augmenter=None):
    """
    Smart balancing: oversample rare classes, undersample frequent classes
    """
    class_counts = Counter(labels)
    current_mean = np.mean(list(class_counts.values()))
    
    if target_mean is None:
        target_mean = current_mean
    
    print("\n" + "="*60)
    print("📊 SMART CLASS BALANCING")
    print("="*60)
    print(f"Current mean: {current_mean:.1f}")
    print(f"Target mean: {target_mean:.1f}")
    print(f"Oversample threshold: < {target_mean * oversample_threshold:.1f}")
    print(f"Undersample threshold: > {target_mean * undersample_threshold:.1f}")
    
    balanced_data = []
    balanced_labels = []
    
    stats = {
        'oversampled': [],
        'undersampled': [],
        'kept': []
    }
    
    for class_label, current_count in class_counts.items():
        # Get samples of this class
        class_indices = [i for i, lbl in enumerate(labels) if lbl == class_label]
        class_samples = [data[i] for i in class_indices]
        
        # Decide strategy
        if current_count < target_mean * oversample_threshold:
            # Oversample rare class
            target_count = int(target_mean)
            needed = target_count - current_count
            
            if needed > 0:
                oversample_indices = np.random.choice(len(class_samples), needed, replace=True)
                oversampled_samples = [class_samples[i] for i in oversample_indices]
                
                balanced_data.extend(class_samples)
                balanced_data.extend(oversampled_samples)
                balanced_labels.extend([class_label] * current_count)
                balanced_labels.extend([class_label] * needed)
                
                stats['oversampled'].append((class_label, current_count, target_count, needed))
                
        elif current_count > target_mean * undersample_threshold:
            # Undersample frequent class
            target_count = int(target_mean * 1.2)
            keep_indices = np.random.choice(current_count, min(target_count, current_count), replace=False)
            kept_samples = [class_samples[i] for i in keep_indices]
            
            balanced_data.extend(kept_samples)
            balanced_labels.extend([class_label] * len(kept_samples))
            
            stats['undersampled'].append((class_label, current_count, len(kept_samples), current_count - len(kept_samples)))
        else:
            # Keep as is
            balanced_data.extend(class_samples)
            balanced_labels.extend([class_label] * current_count)
            stats['kept'].append((class_label, current_count))
    
    balanced_data = np.array(balanced_data)
    balanced_labels = np.array(balanced_labels)
    
    print(f"\n📊 Balancing Results:")
    print(f"   Oversampled classes: {len(stats['oversampled'])}")
    for cls, orig, new, added in stats['oversampled'][:10]:
        print(f"      {cls}: {orig} → {new} (+{added})")
    
    print(f"\n   Undersampled classes: {len(stats['undersampled'])}")
    for cls, orig, new, removed in stats['undersampled'][:10]:
        print(f"      {cls}: {orig} → {new} (-{removed})")
    
    print(f"\n   Kept classes: {len(stats['kept'])}")
    
    print(f"\n✅ Smart balancing complete!")
    print(f"   Original samples: {len(data):,}")
    print(f"   Balanced samples: {len(balanced_data):,}")
    
    return balanced_data, balanced_labels, stats

# ============================================================
# 6. VISUALIZE BALANCING RESULTS
# ============================================================

def visualize_balancing(original_labels, balanced_labels):
    """
    Visualize class distribution before and after balancing
    """
    original_counts = Counter(original_labels)
    balanced_counts = Counter(balanced_labels)
    
    # Get common classes
    all_classes = sorted(set(original_counts.keys()) | set(balanced_counts.keys()))
    original_dist = [original_counts.get(cls, 0) for cls in all_classes]
    balanced_dist = [balanced_counts.get(cls, 0) for cls in all_classes]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Before balancing
    axes[0].hist(list(original_counts.values()), bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Samples per Class')
    axes[0].set_ylabel('Number of Classes')
    axes[0].set_title(f'Before Balancing\nMin={min(original_counts.values())}, Max={max(original_counts.values())}, Mean={np.mean(list(original_counts.values())):.1f}')
    axes[0].grid(True, alpha=0.3)
    
    # After balancing
    axes[1].hist(list(balanced_counts.values()), bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Samples per Class')
    axes[1].set_ylabel('Number of Classes')
    axes[1].set_title(f'After Balancing\nMin={min(balanced_counts.values())}, Max={max(balanced_counts.values())}, Mean={np.mean(list(balanced_counts.values())):.1f}')
    axes[1].grid(True, alpha=0.3)
    
    # Comparison (sorted by original)
    sorted_indices = np.argsort(original_dist)[::-1][:50]
    x = range(len(sorted_indices))
    
    axes[2].bar(x, [original_dist[i] for i in sorted_indices], alpha=0.7, label='Original', color='red')
    axes[2].bar(x, [balanced_dist[i] for i in sorted_indices], alpha=0.7, label='Balanced', color='green')
    axes[2].set_xlabel('Class Index')
    axes[2].set_ylabel('Number of Samples')
    axes[2].set_title('Top 50 Classes Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('class_balancing_results.png', dpi=150)
    plt.show()
    print("📊 Saved: class_balancing_results.png")

# ============================================================
# 7. CREATE BALANCED DATALOADER FOR TRAINING
# ============================================================

def create_balanced_dataloader(data, labels, batch_size=32, use_weighted_sampler=True):
    """
    Create a dataloader with balanced sampling
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dataset
    dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).long())
    
    if use_weighted_sampler:
        # Create weighted sampler
        class_counts = Counter(labels)
        class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
        sample_weights = np.array([class_weights[label] for label in labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        print("✅ Using WeightedRandomSampler for balanced batches")
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        print("✅ Using standard shuffling")
    
    return dataloader

