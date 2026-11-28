import numpy as np

# helper function to normalize a vector
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n

def project_points_orthographic(points, direction, up, world_center, world_size, image_shape):
    H, W = image_shape

    # create vectors for each camera axis
    w = normalize(direction)        # viewing light direction
    u = normalize(np.cross(up, w))  # horizontal axis of image
    v = normalize(np.cross(w, u))   # vertical axis of image

    # center the sculpture at the origin
    rel = points - world_center[None, :]

    # compute how far the voxel is along each camera axis
    x = rel @ u
    y = rel @ v
    z = rel @ w

    # convert world coordinates to pixel coordinates
    half = world_size / 2.0
    px = ((x + half) / world_size) * (W - 1)
    py = ((half - y) / world_size) * (H - 1)

    # find if projection lies inside the image
    valid = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)

    return px, py, valid, z


def project_pixels(points, source):
    x, y, valid, depth = project_points_orthographic(
        points, source.direction, source.up, source.world_center,
        source.world_size, source.image.shape,
    )
    return np.rint(y).astype(np.int32), np.rint(x).astype(np.int32), valid, depth


def pixel_indices(points, source):
    """Flatten pixel coordinates, using -1 for points outside the frame."""
    rows, cols, valid, _ = project_pixels(points, source)
    return np.where(valid, rows * source.image.shape[1] + cols, -1)


def projected_mask(indices, shape):
    mask = np.zeros(np.prod(shape), dtype=bool)
    mask[indices[indices >= 0]] = True
    return mask.reshape(shape)
