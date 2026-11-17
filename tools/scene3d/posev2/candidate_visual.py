"""Pixel measurements for the candidate-only Blender silhouette passes."""
import numpy as np


def measure(body, visible, frame_box, threshold=128):
    """Measure expanded body silhouette and depth-tested visible mask at original pixel scale.

    frame_box = (left, top, right, bottom), exclusive, on the expanded canvas.
    Truncation at the expanded canvas is an error, not a complete silhouette.
    """
    body, visible = np.asarray(body), np.asarray(visible)
    if body.ndim != 2 or visible.shape != body.shape:
        raise ValueError('silhouette masks must have the same two-dimensional shape')
    h, w = body.shape
    x0, y0, x1, y1 = map(int, frame_box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('invalid frame box')
    sil, vis = body >= threshold, visible >= threshold
    ys, xs = np.nonzero(sil)
    if not len(xs):
        raise ValueError('empty body silhouette')
    if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
        raise ValueError('body silhouette exceeds expanded canvas')
    inside = sil[y0:y1, x0:x1]
    count = int(inside.sum())
    visible_count = int((inside & vis[y0:y1, x0:x1]).sum())
    edges = []
    if xs.min() < x0: edges.append('left')
    if xs.max() >= x1: edges.append('right')
    if ys.min() < y0: edges.append('top')
    if ys.max() >= y1: edges.append('bottom')
    return {'height_fraction': float((ys.max() - ys.min() + 1) / (y1 - y0)),
            'cropped_edges': edges,
            'visible_fraction': float(visible_count / count) if count else 0.0,
            'body_pixels_in_frame': count, 'visible_pixels': visible_count,
            'body_pixels_total': int(sil.sum()), 'method': 'blender_body_and_depth_tested_masks'}
