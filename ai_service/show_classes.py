import numpy as np
import pandas as pd
from pathlib import Path

# Path to your label encoder
LABEL_PATH = Path("InferenceLLM/models_features/label_encoder.npy")

print("=" * 60)
print("ASL MODEL CLASSES - EXPORT TO CSV")
print("=" * 60)

# Load the label encoder
try:
    LABELS_DATA = np.load(LABEL_PATH, allow_pickle=True)
    print(f"\n✅ Loaded label file from: {LABEL_PATH}")
    
    # Handle different formats
    if isinstance(LABELS_DATA, np.ndarray) and LABELS_DATA.ndim == 0:
        LABELS_DATA = LABELS_DATA.item()
    
    # Create DataFrame
    if isinstance(LABELS_DATA, dict):
        # Dictionary format: {class_name: class_id}
        print(f"📊 Format: Dictionary")
        print(f"Total classes: {len(LABELS_DATA)}")
        
        # Create list of (class_id, class_name)
        data = []
        for class_name, class_id in LABELS_DATA.items():
            data.append({
                'class_id': class_id,
                'class_name': class_name,
                'first_letter': class_name[0] if class_name else '',
                'length': len(class_name),
                'word_count': len(class_name.split())
            })
        
        # Sort by class_id
        data.sort(key=lambda x: x['class_id'])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
    elif isinstance(LABELS_DATA, (list, np.ndarray)):
        # List/Array format
        print(f"📊 Format: List/Array")
        print(f"Total classes: {len(LABELS_DATA)}")
        
        data = []
        for class_id, class_name in enumerate(LABELS_DATA):
            data.append({
                'class_id': class_id,
                'class_name': class_name,
                'first_letter': class_name[0] if class_name else '',
                'length': len(class_name),
                'word_count': len(class_name.split())
            })
        
        df = pd.DataFrame(data)
    
    else:
        print(f"❌ Unknown format: {type(LABELS_DATA)}")
        sys.exit(1)
    
    # Save to CSV
    output_path = Path("asl_classes_export.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path.absolute()}")
    
    # Also save a more detailed version with grouping
    output_detailed = Path("asl_classes_detailed.csv")
    
    # Add grouping for analysis
    df['starts_with_vowel'] = df['class_name'].str.match(r'^[AEIOUaeiou]', na=False)
    df['ends_with_ing'] = df['class_name'].str.endswith('ING', na=False)
    df['has_space'] = df['class_name'].str.contains(' ', na=False)
    df['has_number'] = df['class_name'].str.contains(r'\d', na=False)
    
    df.to_csv(output_detailed, index=False)
    print(f"✅ Saved detailed version to: {output_detailed.absolute()}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total classes: {len(df)}")
    print(f"Unique first letters: {df['first_letter'].nunique()}")
    print(f"Average name length: {df['length'].mean():.1f} characters")
    print(f"Multi-word phrases: {df['word_count'].gt(1).sum()}")
    print(f"Start with vowel: {df['starts_with_vowel'].sum()}")
    print(f"End with 'ING': {df['ends_with_ing'].sum()}")
    print(f"Contains numbers: {df['has_number'].sum()}")
    
    # Show first 20 rows
    print("\n" + "=" * 60)
    print("FIRST 20 CLASSES")
    print("=" * 60)
    print(df[['class_id', 'class_name']].head(20).to_string(index=False))
    
    # Show last 20 rows
    print("\n" + "=" * 60)
    print("LAST 20 CLASSES")
    print("=" * 60)
    print(df[['class_id', 'class_name']].tail(20).to_string(index=False))
    
    # Find potential confusing pairs
    print("\n" + "=" * 60)
    print("POTENTIALLY CONFUSING SIGN PAIRS")
    print("=" * 60)
    
    # Group by first 3 letters
    df['first_3'] = df['class_name'].str[:3].str.upper()
    duplicate_starts = df[df.duplicated(subset=['first_3'], keep=False)]
    
    if len(duplicate_starts) > 0:
        print("\n⚠️ Signs with same first 3 letters (may look similar):")
        for first3, group in duplicate_starts.groupby('first_3'):
            if len(group) > 1:
                print(f"\n  '{first3}*' ({len(group)} signs):")
                for _, row in group.iterrows():
                    print(f"    - {row['class_name']}")
    else:
        print("No obvious confusing pairs found by first 3 letters")
    
    # Group by length and first letter
    df['first_letter_len'] = df['first_letter'] + '_' + df['length'].astype(str)
    similar_by_len = df[df.duplicated(subset=['first_letter_len'], keep=False)]
    
    if len(similar_by_len) > 0:
        print("\n⚠️ Signs with same first letter and same length:")
        for key, group in similar_by_len.groupby('first_letter_len'):
            if len(group) > 1:
                print(f"\n  Group: {key} ({len(group)} signs):")
                for _, row in group.iterrows():
                    print(f"    - {row['class_name']}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Try alternative location
    alt_path = Path("InferenceLLM/models/label_encoder.npy")
    if alt_path.exists():
        print(f"\nTrying alternative path: {alt_path}")
        LABELS_DATA = np.load(alt_path, allow_pickle=True)
        if isinstance(LABELS_DATA, dict):
            data = []
            for class_name, class_id in LABELS_DATA.items():
                data.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'first_letter': class_name[0] if class_name else '',
                    'length': len(class_name)
                })
            data.sort(key=lambda x: x['class_id'])
            df = pd.DataFrame(data)
            df.to_csv("asl_classes_export.csv", index=False)
            print(f"✅ Saved to asl_classes_export.csv")
    else:
        print("❌ No label file found")