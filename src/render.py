from dataclasses import replace

import numpy as np
from projections import pixel_indices, project_points_orthographic, projected_mask


def doubled_shadow_source(source):
    return replace(source, image=source.image.repeat(2, axis=0).repeat(2, axis=1))

def render_shadow(voxels, voxel_centers, src):
    return projected_mask(pixel_indices(voxel_centers[voxels], src), src.image.shape)


def render_mesh_shadow(triangles, source, *, cell_centers=False):
    """Rasterize projected STL triangles at the target image's pixel centers."""
    height, width = source.image.shape
    x, y, _, _ = project_points_orthographic(
        triangles.reshape(-1, 3), source.direction, source.up,
        source.world_center, source.world_size, source.image.shape,
    )
    if cell_centers:
        if width > 1:
            x = x * (width / (width - 1)) - .5
        if height > 1:
            y = y * (height / (height - 1)) - .5
    x, y = x.reshape(-1, 3), y.reshape(-1, 3)
    tolerance = 1e-5
    left = np.maximum(0, np.ceil(x.min(axis=1) - tolerance)).astype(int)
    right = np.minimum(width - 1, np.floor(x.max(axis=1) + tolerance)).astype(int)
    top = np.maximum(0, np.ceil(y.min(axis=1) - tolerance)).astype(int)
    bottom = np.minimum(height - 1, np.floor(y.max(axis=1) + tolerance)).astype(int)
    ax, bx, cx = x.T
    ay, by, cy = y.T
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    drawable = (left <= right) & (top <= bottom) & (np.abs(area) > 1e-9)
    shadow = np.zeros((height, width), dtype=bool)
    if not drawable.any():
        return shadow
    ax, bx, cx, ay, by, cy, left, right, top, bottom, area = (
        data[drawable] for data in (ax, bx, cx, ay, by, cy, left, right, top, bottom, area)
    )
    for dx in range((right - left).max() + 1):
        for dy in range((bottom - top).max() + 1):
            col, row = left + dx, top + dy
            u = ((col - ax) * (cy - ay) - (row - ay) * (cx - ax)) / area
            v = ((bx - ax) * (row - ay) - (by - ay) * (col - ax)) / area
            hit = (col <= right) & (row <= bottom) & (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1 + tolerance)
            shadow[row[hit], col[hit]] = True
    return shadow
