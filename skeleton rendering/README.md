# Sign Language Skeleton Extraction & Visualization Pipeline

This task provides a full pipeline to extract, preprocess, and visualize human pose and hand landmarks using MediaPipe Holistic.

---

## 🚀 Pipeline Overview

1. Landmark Extraction  
2. Preprocessing  
3. Skeleton Rendering  
4. Comparison Visualization  

---

## 📂 Project Structure
```bash
.
├── videos/
├── mediapipe_landmarks/
├── preprocessed_mediapipe_landmarks/
├── raw_mediapipe_rendered_videos/
├── preprocessed_rendered_videos/
├── comparison_videos/

├── extract_mediapipe_holistic_landmarks.py
├── render_all.py
├── visualization.py
```

---

## 🧠 Landmark Extraction

Script: `extract_mediapipe_holistic_landmarks.py`

- Uses MediaPipe Holistic to extract:
  - Pose (33 landmarks × 4)
  - Face (60 landmarks)
  - Left hand (21 landmarks)
  - Right hand (21 landmarks)

- Each frame → 438 features

**Output:**  
`.npy` file per video with shape `(num_frames, 438)`

---

## 🛠️ Preprocessing

Script:  
`Preprocessing_Landmarks/scripts/preprocess_v3.py`

Features:
- Frame normalization  
- Temporal smoothing  
- Noise reduction  

---

## 🎨 Skeleton Rendering

Script: `render_all.py`

- Converts `.npy` → `.mp4`
- Features:
  - Dynamic scaling  
  - Pose + hands rendering  
  - Color coding:
    - Green → Pose  
    - Blue → Left hand  
    - Red → Right hand  
  - Legend overlay  

---

## 📊 Visualization

Script: `visualization.py`

Creates comparison videos showing:
- Original video  
- Raw skeleton  
- Preprocessed skeleton  

---

## ▶️ How to Run

### 1. Extract landmarks
```bash
python extract_mediapipe_holistic_landmarks.py
```
### 2. Preprocess landmarks
```bash
python Preprocessing_Landmarks/scripts/preprocess_v3.py --input-dir mediapipe_landmarks --output-dir preprocessed_mediapipe_landmarks --target-frames 96 --smooth
```
### 3. Render raw skeletons
```bash
python render_all.py --input mediapipe_landmarks --output raw_mediapipe_rendered_videos
```
### 4. Render processed skeletons
```bash
python render_all.py --input preprocessed_mediapipe_landmarks --output preprocessed_rendered_videos
```
### 5. Generate comparison videos
```bash
python visualization.py --original_dir "videos" --raw_dir "raw_mediapipe_rendered_videos" --proc_dir "preprocessed_rendered_videos" --output_dir "comparison_videos"
```

## 🧩 Dependencies
```bash
pip install opencv-python mediapipe numpy
```