
---

## Folder Structure

| Folder/File | Description |
|-------------|-------------|
| **Analysis** | Scripts and notebooks for analyzing skeleton data and visualizations. |
| **Models** | Trained models (if any) for preprocessing or classification tasks. |
| **Preprocessing_Landmarks** | Scripts for extracting and preprocessing skeleton landmarks from videos. |
| **skeleton rendering** | Scripts for rendering skeletons from preprocessed landmarks, including comparison visualizations. |
| **split_data.py** | Python script to split datasets into train/validation/test sets. |


---

## Cases Overview

| Case | Description | Kaggle Dataset |
|------|-------------|----------------|
| **Case 1** | Not cleaned only | – |
| **Case 2** | Not cleaned + preprocessing | [Dataset Link](https://www.kaggle.com/datasets/mariamhany44/100-words-preprocessed-not-cleaned-no-mask/data) |
| **Case 3** | Not cleaned + preprocessing + mask | – |
| **Case 4** | Clean only | [Dataset Link](https://www.kaggle.com/datasets/mariamhany44/100words-cleaned-not-preprocessed-no-mask/data) |
| **Case 5** | Clean + preprocessing | [Dataset Link](https://www.kaggle.com/datasets/mariamhany44/100-words-preprocessed-landmarks-cleaned-no-mask/data) |
| **Case 6** | Clean + preprocessing + mask | – |
 

---

## Features

- Render skeleton landmarks on videos using **MediaPipe Holistic**.
- Compare original vs processed videos side by side.
- Flexible input/output directory handling for batches of videos.
- Adjustable visualization parameters:
  - `--delay`: Frame display delay.
  - `--scale`: Resize output for easier viewing.
- Supports batch processing (`batch-1` to `batch-5` and `real-test-batch`).  

---
