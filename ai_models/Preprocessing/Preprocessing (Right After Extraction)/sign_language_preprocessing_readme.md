# 📘 Sign Language Keypoint Preprocessing Pipeline

This repository contains a complete preprocessing pipeline designed for **Sign Language Recognition (SLR)** and **Sign Language Translation (SLT)** systems. It transforms raw, noisy Mediapipe keypoints into clean, aligned, and model-ready sequences.

The pipeline provides:
- Intelligent hand reconstruction
- Wrist-relative gap filling
- Hand-swap correction
- Outlier removal
- Frame alignment
- Feature concatenation
- Optional data augmentation
- Model-ready tensor output

---

## 🧩 1. Input Format
Your dataset consists of four Mediapipe keypoint streams:

| Stream | Shape | Description |
|--------|--------|-------------|
| **Pose** | (num_videos, num_frames, 132) | 44 landmarks × (x, y, z) |
| **Left Hand** | (num_videos, num_frames, 63) | 21 LH landmarks × 3 |
| **Right Hand** | (num_videos, num_frames, 63) | 21 RH landmarks × 3 |
| **Face** | (num_videos, num_frames, 180) | 60 facial keypoints × 3 |

These streams often contain:
- Missing hand frames
- Swapped left/right hands
- Sudden jumps or jitter
- False detections
- Videos with inconsistent frame lengths

---

## 🖐 2. Wrist Extraction
Using the pose stream:
- **Left Wrist** → landmark index `15`
- **Right Wrist** → landmark index `16`

Wrist coordinates are used as anchors for hand reconstruction.

---

## 🧠 3. Intelligent Hand Reconstruction
This is the most important stage. Hand detections are cleaned and reconstructed using several advanced steps.

### ✔ 3.1 Swap Detection & Correction
Hands are auto-swapped when:
- The left hand is closer to the right wrist
- The right hand is closer to the left wrist

This fixes Mediapipe’s common LH ↔ RH confusion.

### ✔ 3.2 Wrist-Distance Gating
If a hand is too far from the wrist, it is considered invalid and reset to zero.

### ✔ 3.3 Gap Classification
Before filling missing frames, gaps are categorized:

| Gap Type | Length | Strategy |
|----------|---------|----------|
| **Small** | ≤ 6 frames | Interpolate or carry-forward (wrist-relative) |
| **Medium** | 7–15 frames | Wrist-relative carry-forward only |
| **Large** | > 15 frames | Leave missing (avoid fake hallucination) |

### ✔ 3.4 Wrist-Relative Reconstruction
Instead of interpolating absolute landmark coordinates, we use:
```
relative = hand - wrist
reconstructed = relative + wrist
```
This creates realistic motion that follows the signer’s body.

---

## 🔧 4. Re-Flattening Hands
After reconstruction:
```
(T, 21, 3) → (T, 63)
```
Hands return to flattened format.

---

## 🧮 5. Feature Concatenation
All streams are combined into a single feature vector per frame:
```
pose   = 132
left   = 63
right  = 63
face   = 180
-----------------------
total  = 438 features
```

Each video becomes:
```
(num_frames, 438)
```

---

## 🔁 6. Output Dataset Structure
Videos are stored in a Python list:
```
[
   video_1 (num_frames, 438),
   video_2 (num_frames, 438),
   ...
]
```
This format integrates naturally with:
- PyTorch DataLoader
- TensorFlow/Keras datasets
- Custom training loops

---

## 🧪 7. Optional Data Augmentation
Supported augmentations:
- Gaussian jitter (noise)
- Minor scaling
- Light rotation
- Time-warping (speed up / slow down)
- Frame dropout simulation (Mediapipe-style)

These augmentations improve robustness and help prevent overfitting.

---

## 📤 8. Final Output
The final output of preprocessing is:
```
processed_videos : List[np.ndarray]
```
Where each video has:
```
shape = (num_frames, 438)
```
The data is fully cleaned, reconstructed, stable, and ready for training with sequential models such as:
- LSTM / GRU
- Transformers
- TCNs (Temporal Convolutional Networks)

---

## 🧠 9. Full Pipeline Summary
1. Load Mediapipe keypoints
2. Extract wrist positions
3. Convert hands to (21,3)
4. Fix swapped hands
5. Gate invalid detections
6. Fill missing frames using the intelligent tiered system
7. Flatten hands
8. Concatenate pose + LH + RH + face
9. Store videos in list form
10. Optionally apply augmentation

The final dataset is consistent, stable, and suitable for training high-performance sign language models.

---

If you'd like, I can also add:
- A pipeline flowchart
- Example visualization (before/after reconstruction)
- A ready-to-use PyTorch Dataset class
- A CLI script to run the entire preprocessing automatically.