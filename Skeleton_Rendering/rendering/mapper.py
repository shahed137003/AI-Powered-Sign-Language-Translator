def make_mapper(xmin, xmax, ymin, ymax, W, H, margin=20):
    bw = max(xmax - xmin, 1e-6)
    bh = max(ymax - ymin, 1e-6)

    usable_w = max(W - 2 * margin, 10)
    usable_h = max(H - 2 * margin, 10)

    s = min(usable_w / bw, usable_h / bh)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    tx, ty = W / 2.0 - s * cx, H / 2.0 - s * cy

    def map_xy(x, y):
        return int(s * x + tx), int(s * y + ty)

    return map_xy
