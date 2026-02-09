# Preprocessing_Landmarks

This package preprocesses MediaPipe keypoints saved as `*.npy` arrays of shape **(T, 438)** using the layout:

- **Pose:** 33 landmarks × (x, y, z, vis) = 132
- **Face:** 60 landmarks × (x, y, z) = 180
- **Left hand:** 21 landmarks × (x, y, z) = 63
- **Right hand:** 21 landmarks × (x, y, z) = 63
- **Total:** 438

It’s built for sign-language recognition pipelines (GCN/GCNN, Transformers, etc.) where you want:
- consistent normalization across frames inside a clip
- safer hands (swap-fix + wrist-distance gating + conservative gap fill)
- optional One-Euro smoothing to reduce jitter

---

## Output format

The output is a `*.npy` array with:
- **shape:** (T, 438)
- **dtype:** float32

After preprocessing, coordinates are **root-centered and scale-normalized**, so values are **not guaranteed to stay in [0, 1]** (that’s expected).

---

## Run (recommended)

### Option A — Run as a module (avoids `ModuleNotFoundError`)
From the repo root (the folder that contains `preprocessing/`):

```powershell
python -m preprocessing.cli_v3 --help
