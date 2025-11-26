
---



```md
# Rendering (robust skeleton videos)

This package renders skeleton videos (`.avi`) from MediaPipe keypoint arrays saved as `*.npy` files of shape **(T, 438)**.

It is **robust** because it does **not** assume x,y are in `[0, 1]`.
Instead, it computes a bounding box across the entire clip and maps coordinates into the output frame with padding.

This makes it useful for both:
- **RAW** keypoints (often in `[0,1]`)
- **PREPROCESSED** keypoints (root+scale normalized; can be negative / > 1)

---

## Output

For each input `file.npy`, it writes:

- `file_robust_skel.avi`

Saved inside the folder specified by `--output-dir`.

---

## Run (PowerShell)

From the repo root (the folder that contains `rendering/`):

```powershell
python scripts/render_skeleton_robust.py `
  --input-dir "--input-dir" `
  --output-dir "--output-dir" `
  --limit 20 `
  --fps 20
