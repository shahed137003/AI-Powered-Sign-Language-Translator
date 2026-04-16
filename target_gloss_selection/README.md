# ASL Gloss Dataset Merging and Cleaning

This project contains a Python-based data processing pipeline designed to aggregate, de-duplicate, and normalize American Sign Language (ASL) glosses from multiple sources. The goal of this project is to create a curated, high-quality list of approximately 573 unique glosses for sign language recognition and research.

## 📋 Project Overview

The script consolidates data from four primary sources:
1.  **Original Dataset (`gloss_counts.csv`):** 147 glosses from our manually cleaned dataset.
2.  **Essential ASL List (`asl_top_250_essential.csv`):** 250 common signs collected from various internet sources.
3.  **Research Dataset (`sorted_mean_rating.csv`):** Academic signs from research papers (initially taking the top 103).
4.  **Additional List (`Additional_Gloss_List.csv`):** Supplemental essential glosses for final expansion.

## 🛠️ Data Processing Workflow

### 1. Initialization and Loading
The script uses `pandas` to load the CSV files, renames various columns (like "Sign" or "gloss") to a standardized **Gloss** header, and selects specific subsets of data to maintain quality control.

### 2. Normalization
To ensure accurate matching across different datasets, all glosses undergo a normalization process:
* **Strip:** Removes leading and trailing whitespace.
* **Lowercase:** Converts all text to lowercase for case-insensitive comparison.

### 3. De-duplication and Overlap Management
The script identifies overlaps between datasets using Python `set` operations to ensure no sign is repeated across different sources.
* **Priority Logic:** The "Original Dataset" is treated as the primary source. If a gloss exists there, it is removed from the other lists.
* **Verification:** The script performs multiple checks before and after merging to confirm a zero-overlap state.

### 4. Merging and Compensation
After resolving overlaps, the datasets are merged. If the total count falls below the target of **500**, the script automatically "compensates" by pulling the next available unique signs from the research paper pool (`sorted_mean_rating.csv`) until the threshold is met.

### 5. Final Cleaning and Formatting
The final stage involves rigorous string cleaning to ensure the dataset is ready for machine learning models:
* **Regex Cleaning:** Removes parenthetical notes (e.g., changing `EAT (FOOD)` to `EAT`).
* **Character Standardization:** Replaces hyphens (`-`) and underscores (`_`) with spaces.
* **Exploding Compound Glosses:** Identifies entries separated by slashes (e.g., `FOOD/EAT`) and splits them into two separate rows.
* **Redundancy Check:** Manually filters out redundant entries (e.g., removing `$5` in favor of `5 DOLLARS`).
* **Standardization:** All final glosses are converted to **UPPERCASE**.

## 📊 Final Output
The pipeline exports a final, cleaned CSV file:
* **Filename:** `573_Gloss_List.csv`
* **Format:** A single-column list of unique, uppercase ASL glosses.

## 🚀 Requirements
To run the script, you will need:
* Python 3.x
* Pandas
* NumPy

```
pip install pandas numpy
```