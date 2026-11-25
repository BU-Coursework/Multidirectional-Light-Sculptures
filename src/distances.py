import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

# Distance fields for binary silhouettes.
def silhouette_distance_fields(mask: np.ndarray):
    """
    Function which computes the distance to background for foreground pixels and
    the distance to foreground for background pixels using Euclidean distance.
    """
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    return dist_inside, dist_outside

def outside_distance(mask: np.ndarray) -> np.ndarray:
    # 0 inside mask, positive outside.
    return distance_transform_edt(~mask) * (~mask)

def inside_distance(mask: np.ndarray) -> np.ndarray:
    # Positive distance inside mask.
    return distance_transform_edt(mask) * mask


def silhouette_weights(mask):
    holes = binary_fill_holes(mask, structure=np.ones((3, 3), dtype=bool)) & ~mask
    thin = mask & (distance_transform_edt(mask) <= 1.5)
    return 1.0 + thin + 2.0 * holes
