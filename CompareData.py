import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA COMPARISON CLASS
# ============================================================================

class DataComparator:
    """
    Compare cleaned vs uncleaned datasets
    """
    
    def __init__(self, uncleaned_data, uncleaned_labels, cleaned_data, cleaned_labels):
        """
        Initialize comparator with both datasets
        
        Args:
            uncleaned_data: numpy array of uncleaned sequences
            uncleaned_labels: numpy array of uncleaned labels
            cleaned_data: numpy array of cleaned sequences
            cleaned_labels: numpy array of cleaned labels
        """
        self.uncleaned_data = uncleaned_data
        self.uncleaned_labels = uncleaned_labels
        self.cleaned_data = cleaned_data
        self.cleaned_labels = cleaned_labels
        
    def compare_basic_stats(self):
        """Compare basic statistics between datasets"""
        print("="*80)
        print("📊 BASIC STATISTICS COMPARISON")
        print("="*80)
        
        stats = {
            'Metric': ['Total Videos', 'Unique Glosses', 'Avg Videos per Gloss', 
                      'Min Videos per Gloss', 'Max Videos per Gloss', 'Median Videos per Gloss'],
            'Uncleaned': [
                len(self.uncleaned_data),
                len(np.unique(self.uncleaned_labels)),
                len(self.uncleaned_data) / len(np.unique(self.uncleaned_labels)),
                min(Counter(self.uncleaned_labels).values()),
                max(Counter(self.uncleaned_labels).values()),
                np.median(list(Counter(self.uncleaned_labels).values()))
            ],
            'Cleaned': [
                len(self.cleaned_data),
                len(np.unique(self.cleaned_labels)),
                len(self.cleaned_data) / len(np.unique(self.cleaned_labels)),
                min(Counter(self.cleaned_labels).values()),
                max(Counter(self.cleaned_labels).values()),
                np.median(list(Counter(self.cleaned_labels).values()))
            ]
        }
        
        df_stats = pd.DataFrame(stats)
        print(df_stats.to_string(index=False))
        
        # Calculate changes
        print("\n📈 Changes:")
        video_change = (len(self.cleaned_data) - len(self.uncleaned_data)) / len(self.uncleaned_data) * 100
        gloss_change = (len(np.unique(self.cleaned_labels)) - len(np.unique(self.uncleaned_labels))) / len(np.unique(self.uncleaned_labels)) * 100
        print(f"  Videos: {len(self.uncleaned_data)} → {len(self.cleaned_data)} ({video_change:+.1f}%)")
        print(f"  Unique glosses: {len(np.unique(self.uncleaned_labels))} → {len(np.unique(self.cleaned_labels))} ({gloss_change:+.1f}%)")
        
        return df_stats
    
    def compare_frame_lengths(self):
        """Compare frame length distributions"""
        print("\n" + "="*80)
        print("📏 FRAME LENGTH COMPARISON")
        print("="*80)
        
        # Calculate frame lengths
        uncleaned_lengths = [len(seq) for seq in self.uncleaned_data]
        cleaned_lengths = [len(seq) for seq in self.cleaned_data]
        
        # Statistics
        stats = {
            'Metric': ['Mean Frames', 'Median Frames', 'Std Frames', 'Min Frames', 'Max Frames', 
                      '10th Percentile', '25th Percentile', '75th Percentile', '90th Percentile'],
            'Uncleaned': [
                np.mean(uncleaned_lengths),
                np.median(uncleaned_lengths),
                np.std(uncleaned_lengths),
                np.min(uncleaned_lengths),
                np.max(uncleaned_lengths),
                np.percentile(uncleaned_lengths, 10),
                np.percentile(uncleaned_lengths, 25),
                np.percentile(uncleaned_lengths, 75),
                np.percentile(uncleaned_lengths, 90)
            ],
            'Cleaned': [
                np.mean(cleaned_lengths),
                np.median(cleaned_lengths),
                np.std(cleaned_lengths),
                np.min(cleaned_lengths),
                np.max(cleaned_lengths),
                np.percentile(cleaned_lengths, 10),
                np.percentile(cleaned_lengths, 25),
                np.percentile(cleaned_lengths, 75),
                np.percentile(cleaned_lengths, 90)
            ]
        }
        
        df_lengths = pd.DataFrame(stats)
        print(df_lengths.to_string(index=False))
        
        # Frame reduction
        avg_reduction = (np.mean(uncleaned_lengths) - np.mean(cleaned_lengths)) / np.mean(uncleaned_lengths) * 100
        print(f"\n📉 Average frame reduction: {avg_reduction:.1f}%")
        
        return uncleaned_lengths, cleaned_lengths
    
    def compare_label_distribution(self):
        """Compare label frequency distributions"""
        print("\n" + "="*80)
        print("🏷️ LABEL DISTRIBUTION COMPARISON")
        print("="*80)
        
        uncleaned_counts = Counter(self.uncleaned_labels)
        cleaned_counts = Counter(self.cleaned_labels)
        
        # Frequency categories
        categories = ['Rare (<5)', 'Low (5-9)', 'Medium (10-49)', 'Frequent (≥50)']
        
        def categorize(counts):
            rare = sum(1 for c in counts.values() if c < 5)
            low = sum(1 for c in counts.values() if 5 <= c < 10)
            medium = sum(1 for c in counts.values() if 10 <= c < 50)
            frequent = sum(1 for c in counts.values() if c >= 50)
            return [rare, low, medium, frequent]
        
        uncleaned_cats = categorize(uncleaned_counts)
        cleaned_cats = categorize(cleaned_counts)
        
        cat_df = pd.DataFrame({
            'Category': categories,
            'Uncleaned': uncleaned_cats,
            'Cleaned': cleaned_cats,
            'Change': [c - u for c, u in zip(cleaned_cats, uncleaned_cats)]
        })
        
        print("\n📊 Label Frequency Categories:")
        print(cat_df.to_string(index=False))
        
        # Top 10 comparison
        print("\n🏆 Top 10 Most Frequent Glosses (Uncleaned):")
        for i, (label, count) in enumerate(uncleaned_counts.most_common(10), 1):
            cleaned_count = cleaned_counts.get(label, 0)
            print(f"  {i:2d}. {label[:35]:<35} {count:>5} → {cleaned_count:>5} videos")
        
        # Labels that were removed
        removed_labels = set(uncleaned_counts.keys()) - set(cleaned_counts.keys())
        if removed_labels:
            print(f"\n❌ Labels removed during cleaning ({len(removed_labels)}):")
            for label in list(removed_labels)[:10]:
                print(f"  - {label}: {uncleaned_counts[label]} videos")
        
        return uncleaned_counts, cleaned_counts
    
    def compare_data_quality(self):
        """Compare data quality metrics"""
        print("\n" + "="*80)
        print("🔍 DATA QUALITY COMPARISON")
        print("="*80)
        
        def check_quality(data):
            issues = {
                'has_nan': 0,
                'has_inf': 0,
                'all_zero': 0,
                'corrupted': 0
            }
            
            for seq in data:
                try:
                    if np.any(np.isnan(seq)):
                        issues['has_nan'] += 1
                    if np.any(np.isinf(seq)):
                        issues['has_inf'] += 1
                    if np.all(seq == 0):
                        issues['all_zero'] += 1
                except:
                    issues['corrupted'] += 1
            
            return issues
        
        uncleaned_issues = check_quality(self.uncleaned_data)
        cleaned_issues = check_quality(self.cleaned_data)
        
        quality_df = pd.DataFrame({
            'Issue': list(uncleaned_issues.keys()),
            'Uncleaned': list(uncleaned_issues.values()),
            'Cleaned': list(cleaned_issues.values()),
            'Fixed': [u - c for u, c in zip(uncleaned_issues.values(), cleaned_issues.values())]
        })
        
        print(quality_df.to_string(index=False))
        
        return uncleaned_issues, cleaned_issues
    
    def compare_memory_usage(self):
        """Compare memory usage"""
        print("\n" + "="*80)
        print("💾 MEMORY USAGE COMPARISON")
        print("="*80)
        
        def get_memory(data):
            total_bytes = 0
            for seq in data:
                total_bytes += seq.nbytes
            return total_bytes / (1024 * 1024)  # MB
        
        uncleaned_mb = get_memory(self.uncleaned_data)
        cleaned_mb = get_memory(self.cleaned_data)
        
        print(f"  Uncleaned: {uncleaned_mb:.2f} MB")
        print(f"  Cleaned: {cleaned_mb:.2f} MB")
        print(f"  Reduction: {(uncleaned_mb - cleaned_mb):.2f} MB ({(1 - cleaned_mb/uncleaned_mb)*100:.1f}%)")
        
        return uncleaned_mb, cleaned_mb
    
    def compare_temporal_stats(self):
        """Compare temporal/motion statistics"""
        print("\n" + "="*80)
        print("⏱️ TEMPORAL STATISTICS COMPARISON")
        print("="*80)
        
        def compute_motion(sequence):
            if len(sequence) <= 1:
                return 0
            differences = np.diff(sequence, axis=0)
            return np.linalg.norm(differences, axis=1).mean()
        
        # Sample for performance
        sample_size = min(500, len(self.uncleaned_data))
        uncleaned_motion = []
        cleaned_motion = []
        
        for i in range(sample_size):
            uncleaned_motion.append(compute_motion(self.uncleaned_data[i]))
            cleaned_motion.append(compute_motion(self.cleaned_data[i]))
        
        print(f"\n📊 Average Motion Energy (based on {sample_size} samples):")
        print(f"  Uncleaned: {np.mean(uncleaned_motion):.4f}")
        print(f"  Cleaned: {np.mean(cleaned_motion):.4f}")
        print(f"  Change: {(np.mean(cleaned_motion) - np.mean(uncleaned_motion)):.4f}")
        
        return uncleaned_motion, cleaned_motion
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        print("\n" + "="*80)
        print("📊 CREATING COMPARISON VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Frame length distribution
        uncleaned_lengths = [len(seq) for seq in self.uncleaned_data]
        cleaned_lengths = [len(seq) for seq in self.cleaned_data]
        
        axes[0, 0].hist(uncleaned_lengths, bins=50, alpha=0.5, label='Uncleaned', color='red')
        axes[0, 0].hist(cleaned_lengths, bins=50, alpha=0.5, label='Cleaned', color='green')
        axes[0, 0].set_xlabel('Number of Frames')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Frame Length Distribution')
        axes[0, 0].legend()
        
        # 2. Label frequency distribution (log scale)
        uncleaned_counts = list(Counter(self.uncleaned_labels).values())
        cleaned_counts = list(Counter(self.cleaned_labels).values())
        
        axes[0, 1].hist(uncleaned_counts, bins=50, alpha=0.5, label='Uncleaned', color='red')
        axes[0, 1].hist(cleaned_counts, bins=50, alpha=0.5, label='Cleaned', color='green')
        axes[0, 1].set_xlabel('Number of Videos per Gloss')
        axes[0, 1].set_ylabel('Number of Glosses')
        axes[0, 1].set_title('Label Frequency Distribution')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        
        # 3. Cumulative distribution
        uncleaned_sorted = np.sort(uncleaned_counts)[::-1]
        cleaned_sorted = np.sort(cleaned_counts)[::-1]
        uncleaned_cumsum = np.cumsum(uncleaned_sorted) / np.sum(uncleaned_sorted)
        cleaned_cumsum = np.cumsum(cleaned_sorted) / np.sum(cleaned_sorted)
        
        axes[0, 2].plot(range(1, len(uncleaned_sorted)+1), uncleaned_cumsum, 'r-', label='Uncleaned')
        axes[0, 2].plot(range(1, len(cleaned_sorted)+1), cleaned_cumsum, 'g-', label='Cleaned')
        axes[0, 2].set_xlabel('Number of Glosses')
        axes[0, 2].set_ylabel('Cumulative Proportion')
        axes[0, 2].set_title('Cumulative Video Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot of frame lengths
        axes[1, 0].boxplot([uncleaned_lengths, cleaned_lengths], labels=['Uncleaned', 'Cleaned'])
        axes[1, 0].set_ylabel('Number of Frames')
        axes[1, 0].set_title('Frame Length Box Plot')
        
        # 5. Label retention pie chart
        common_labels = set(self.uncleaned_labels) & set(self.cleaned_labels)
        removed_labels = set(self.uncleaned_labels) - set(self.cleaned_labels)
        
        axes[1, 1].pie([len(common_labels), len(removed_labels)], 
                      labels=[f'Retained\n({len(common_labels)})', f'Removed\n({len(removed_labels)})'],
                      autopct='%1.1f%%', colors=['green', 'red'])
        axes[1, 1].set_title('Label Retention')
        
        # 6. Video count comparison (top 10)
        uncleaned_top = Counter(self.uncleaned_labels).most_common(10)
        cleaned_top = Counter(self.cleaned_labels).most_common(10)
        
        labels_top = [l for l, _ in uncleaned_top]
        uncleaned_vals = [c for _, c in uncleaned_top]
        cleaned_vals = [Counter(self.cleaned_labels).get(l, 0) for l in labels_top]
        
        x = range(len(labels_top))
        width = 0.35
        axes[1, 2].bar([i - width/2 for i in x], uncleaned_vals, width, label='Uncleaned', color='red')
        axes[1, 2].bar([i + width/2 for i in x], cleaned_vals, width, label='Cleaned', color='green')
        axes[1, 2].set_xlabel('Gloss')
        axes[1, 2].set_ylabel('Number of Videos')
        axes[1, 2].set_title('Top 10 Glosses Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(labels_top, rotation=45, ha='right')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('data_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved as 'data_comparison.png'")
    
    def generate_comparison_report(self):
        """Generate complete comparison report"""
        print("\n" + "="*80)
        print("📝 GENERATING COMPARISON REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("DATA CLEANING COMPARISON REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {pd.Timestamp.now()}")
        
        # Basic stats
        report_lines.append("\n" + "-"*80)
        report_lines.append("1. BASIC STATISTICS")
        report_lines.append("-"*80)
        report_lines.append(f"  Videos - Uncleaned: {len(self.uncleaned_data)}")
        report_lines.append(f"  Videos - Cleaned: {len(self.cleaned_data)}")
        report_lines.append(f"  Change: {len(self.cleaned_data) - len(self.uncleaned_data):+d} ({(len(self.cleaned_data)-len(self.uncleaned_data))/len(self.uncleaned_data)*100:+.1f}%)")
        
        report_lines.append(f"\n  Unique Glosses - Uncleaned: {len(np.unique(self.uncleaned_labels))}")
        report_lines.append(f"  Unique Glosses - Cleaned: {len(np.unique(self.cleaned_labels))}")
        report_lines.append(f"  Change: {len(np.unique(self.cleaned_labels)) - len(np.unique(self.uncleaned_labels)):+d}")
        
        # Frame lengths
        uncleaned_lengths = [len(seq) for seq in self.uncleaned_data]
        cleaned_lengths = [len(seq) for seq in self.cleaned_data]
        
        report_lines.append("\n" + "-"*80)
        report_lines.append("2. FRAME LENGTHS")
        report_lines.append("-"*80)
        report_lines.append(f"  Mean - Uncleaned: {np.mean(uncleaned_lengths):.1f} frames")
        report_lines.append(f"  Mean - Cleaned: {np.mean(cleaned_lengths):.1f} frames")
        report_lines.append(f"  Change: {np.mean(cleaned_lengths) - np.mean(uncleaned_lengths):+.1f} frames")
        
        # Summary
        report_lines.append("\n" + "-"*80)
        report_lines.append("3. SUMMARY OF CHANGES")
        report_lines.append("-"*80)
        
        removed_labels = set(self.uncleaned_labels) - set(self.cleaned_labels)
        if removed_labels:
            report_lines.append(f"  ✓ Removed {len(removed_labels)} rare glosses")
        
        report_lines.append(f"  ✓ Reduced average video length by {np.mean(uncleaned_lengths) - np.mean(cleaned_lengths):.1f} frames")
        
        # Save report
        report_path = Path('cleaning_comparison_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✅ Report saved to {report_path}")
        
        return report_lines
    
    def run_full_comparison(self):
        """Run all comparison methods"""
        print("="*80)
        print("🔄 RUNNING FULL DATA COMPARISON")
        print("="*80)
        
        # Run all comparisons
        basic_stats = self.compare_basic_stats()
        uncleaned_lengths, cleaned_lengths = self.compare_frame_lengths()
        uncleaned_counts, cleaned_counts = self.compare_label_distribution()
        uncleaned_issues, cleaned_issues = self.compare_data_quality()
        uncleaned_mem, cleaned_mem = self.compare_memory_usage()
        uncleaned_motion, cleaned_motion = self.compare_temporal_stats()
        self.create_visualizations()
        self.generate_comparison_report()
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE!")
        print("="*80)
        
        return {
            'basic_stats': basic_stats,
            'frame_lengths': {'uncleaned': uncleaned_lengths, 'cleaned': cleaned_lengths},
            'label_counts': {'uncleaned': uncleaned_counts, 'cleaned': cleaned_counts},
            'quality_issues': {'uncleaned': uncleaned_issues, 'cleaned': cleaned_issues},
            'memory': {'uncleaned': uncleaned_mem, 'cleaned': cleaned_mem},
            'motion': {'uncleaned': uncleaned_motion, 'cleaned': cleaned_motion}
        }

