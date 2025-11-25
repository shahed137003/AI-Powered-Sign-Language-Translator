# WLASL Landmarks

**Dataset Type:** Preprocessed Landmarks from WLASL Videos

---

## 📁 Description

This dataset contains **preprocessed landmarks extracted from WLASL (Word-Level American Sign Language) videos**. Each video is represented by **four landmark groups**:

| Landmark Group | Description                         |
| -------------- | ----------------------------------- |
| Right Hand     | Keypoints of the right hand         |
| Left Hand      | Keypoints of the left hand          |
| Pose           | Body skeleton keypoints             |
| Face           | Keypoints of lips and eyebrows only |

* **Format:** `.npy` files (NumPy arrays)
* **Annotation:** Each file is mapped to its **corresponding sign phrase**, and the filename **matches the phrase**
* Captures hand gestures, body posture, and facial expressions for **accurate sign recognition**

---

## 🔗 Download

You can download the dataset here:

[Google Drive - WLASL Landmarks Dataset](https://drive.google.com/file/d/13nd9qCfoYUNdNMGytkd5PoV32G4iL_e3/view?usp=sharing)

> ⚠️ Note: Raw video data is **not included** due to size constraints. Only preprocessed landmarks are provided.

---

## 📌 Usage

Load landmarks in Python:

```python
import numpy as np

landmark_file = "hello.npy"
right_hand, left_hand, pose, face = np.load(landmark_file, allow_pickle=True)

print("Right hand shape:", right_hand.shape)
print("Left hand shape:", left_hand.shape)
print("Pose shape:", pose.shape)
print("Face shape:", face.shape)  # lips & eyebrows only
```

* Each file contains **all four landmark groups** for full gesture representation.
* These landmarks can be directly used to **train or evaluate AI models** for sign language recognition.

---

## ⚠️ Notes

* Face landmarks include **lips and eyebrows only**.
* All landmarks are normalized per video to handle scale and position differences.
* Ensure **NumPy >= 1.24** and **Python >= 3.10** for compatibility.

---

## 📄 Citation

If you use this dataset, please cite WLASL:

> [WLASL: Word-Level American Sign Language Dataset](https://dxli94.github.io/WLASL/)

---
