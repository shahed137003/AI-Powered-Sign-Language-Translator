import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

class SignLanguageAnalyzer:
    """
    Flexible analyzer for sign language landmark data
    """
    
    def __init__(self, output_dir="D:\\GP\\DataAnalysis", min_frames=10, max_frames=500):
        """
        Initialize the analyzer with custom settings
        
        Parameters:
        - output_dir: Directory to save results
        - min_frames: Minimum frames threshold
        - max_frames: Maximum frames threshold
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_frames = min_frames
        self.max_frames = max_frames
        
    def analyze_label_distribution(self, labels):
        """Analyze label distribution"""
        label_counter = Counter(labels)
        label_df = pd.DataFrame({
            'label': list(label_counter.keys()),
            'count': list(label_counter.values())
        }).sort_values('count', ascending=False)
        
        return label_counter, label_df
    
    def analyze_frame_lengths(self, data, labels):
        """Analyze frame length distribution"""
        frame_lengths = np.array([len(seq) for seq in data])
        
        stats = {
            'min': frame_lengths.min(),
            'max': frame_lengths.max(),
            'mean': frame_lengths.mean(),
            'median': np.median(frame_lengths),
            'std': frame_lengths.std(),
            'percentiles': {
                p: np.percentile(frame_lengths, p) 
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        return frame_lengths, stats
    
    def analyze_data_quality(self, data, labels):
        """Analyze data quality issues"""
        quality_issues = {
            'has_nan': 0,
            'has_inf': 0,
            'all_zero': 0,
            'corrupted': 0
        }
        
        for i, seq in enumerate(data):
            try:
                if np.any(np.isnan(seq)):
                    quality_issues['has_nan'] += 1
                if np.any(np.isinf(seq)):
                    quality_issues['has_inf'] += 1
                if np.all(seq == 0):
                    quality_issues['all_zero'] += 1
            except:
                quality_issues['corrupted'] += 1
        
        return quality_issues
    
    def analyze_alphabet_distribution(self, labels):
        """Analyze distribution by first letter"""
        label_counter = Counter(labels)
        letter_stats = defaultdict(lambda: {'count': 0, 'unique_labels': set()})
        
        for label, count in label_counter.items():
            cleaned = re.sub(r'^\d+\s+', '', label)
            if cleaned and cleaned[0].isalpha():
                letter = cleaned[0].upper()
            elif label and label[0].isalpha():
                letter = label[0].upper()
            else:
                letter = '#'
            
            letter_stats[letter]['count'] += count
            letter_stats[letter]['unique_labels'].add(label)
        
        return letter_stats
    
    def create_visualizations(self, label_counter, frame_lengths, save=True):
        """Create basic visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Label distribution
        top_labels = dict(sorted(label_counter.items(), key=lambda x: x[1], reverse=True)[:20])
        axes[0, 0].barh(list(top_labels.keys()), list(top_labels.values()))
        axes[0, 0].set_title('Top 20 Most Frequent Glosses')
        axes[0, 0].invert_yaxis()
        
        # Frame length histogram
        axes[0, 1].hist(frame_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Number of Frames')
        axes[0, 1].set_ylabel('Number of Videos')
        axes[0, 1].set_title('Frame Length Distribution')
        axes[0, 1].axvline(np.median(frame_lengths), color='r', linestyle='--', label=f'Median: {np.median(frame_lengths):.0f}')
        axes[0, 1].legend()
        
        # Cumulative distribution
        counts_array = list(label_counter.values())
        sorted_counts = np.sort(counts_array)[::-1]
        cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        axes[1, 0].plot(range(1, len(sorted_counts)+1), cumulative)
        axes[1, 0].set_xlabel('Number of Glosses')
        axes[1, 0].set_ylabel('Cumulative Proportion')
        axes[1, 0].set_title('Cumulative Video Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Frequency categories pie chart
        rare = sum(1 for c in counts_array if c < 5)
        low = sum(1 for c in counts_array if 5 <= c < 10)
        medium = sum(1 for c in counts_array if 10 <= c < 50)
        frequent = sum(1 for c in counts_array if c >= 50)
        
        sizes = [rare, low, medium, frequent]
        labels_pie = ['Rare (<5)', 'Low (5-9)', 'Medium (10-49)', 'Frequent (≥50)']
        axes[1, 1].pie(sizes, labels=labels_pie, autopct='%1.1f%%')
        axes[1, 1].set_title('Gloss Frequency Categories')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'analysis_visualization.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, data, labels):
        """Generate a complete analysis report"""
        print("="*70)
        print("📊 SIGN LANGUAGE DATA ANALYSIS REPORT")
        print("="*70)
        
        # Basic stats
        label_counter, label_df = self.analyze_label_distribution(labels)
        frame_lengths, frame_stats = self.analyze_frame_lengths(data, labels)
        quality_issues = self.analyze_data_quality(data, labels)
        letter_stats = self.analyze_alphabet_distribution(labels)
        
        print(f"\n📈 Basic Statistics:")
        print(f"  Total videos: {len(data)}")
        print(f"  Unique glosses: {len(label_counter)}")
        print(f"  Average videos/gloss: {len(data)/len(label_counter):.2f}")
        
        print(f"\n📏 Frame Length Statistics:")
        for key, value in frame_stats.items():
            if key != 'percentiles':
                print(f"  {key.capitalize()}: {value:.2f}" if isinstance(value, float) else f"  {key.capitalize()}: {value}")
        
        print(f"\n🔍 Quality Issues:")
        for issue, count in quality_issues.items():
            print(f"  {issue.replace('_', ' ').capitalize()}: {count} ({count/len(data)*100:.2f}%)")
        
        print(f"\n🔤 Alphabet Distribution (Top 10):")
        letter_df = pd.DataFrame([
            {'letter': k, 'total_videos': v['count'], 'unique_labels': len(v['unique_labels'])}
            for k, v in letter_stats.items() if k != '#'
        ]).sort_values('total_videos', ascending=False)
        print(letter_df.head(10).to_string(index=False))
        
        # Create visualizations
        self.create_visualizations(label_counter, frame_lengths)
        
        return {
            'label_distribution': label_df,
            'frame_stats': frame_stats,
            'quality_issues': quality_issues,
            'letter_stats': letter_stats
        }
    
    def quick_analysis(self, data, labels):
        """Quick analysis without saving files"""
        label_counter = Counter(labels)
        
        print(f"✅ Quick Analysis Results:")
        print(f"  • {len(data)} total videos")
        print(f"  • {len(label_counter)} unique glosses")
        print(f"  • Average length: {np.mean([len(seq) for seq in data]):.1f} frames")
        print(f"  • Most common: {label_counter.most_common(1)[0][0]} ({label_counter.most_common(1)[0][1]} videos)")
        print(f"  • Rarest: {label_counter.most_common()[-1][0]} ({label_counter.most_common()[-1][1]} videos)")
        
        return label_counter

# Convenience function for quick use
def quick_analyze(data, labels):
    """Quick analysis function"""
    analyzer = SignLanguageAnalyzer()
    return analyzer.quick_analysis(data, labels)