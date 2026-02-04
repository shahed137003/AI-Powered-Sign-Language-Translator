import numpy as np

from .layout_438 import unpack_frame


# ---------- bbox + mapping for arbitrary coords ----------
def _valid_mask_xy(arr_xy, eps=1e-8):
    finite = np.isfinite(arr_xy).all(axis=1)
    nonzero = np.any(np.abs(arr_xy) > eps, axis=1)
    return finite & nonzero


def compute_bbox_for_clip(sample, eps=1e-8):
    xs, ys = [], []
    T = sample.shape[0]
    for t in range(T):
        pose, face, lh, rh = unpack_frame(sample[t])
        for arr in (pose[:, :2], face[:, :2], lh[:, :2], rh[:, :2]):
            m = _valid_mask_xy(arr, eps=eps)
            if np.any(m):
                xs.append(arr[m, 0])
                ys.append(arr[m, 1])
    if not xs:
        return -1.0, 1.0, -1.0, 1.0

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    pad = 0.1
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    return xmin, xmax, ymin, ymax
