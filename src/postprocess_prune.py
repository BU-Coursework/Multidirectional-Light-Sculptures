import numpy as np
from scipy import ndimage

from projections import project_pixels
from shadow_hull import FACE_NEIGHBORS, connected_components_3d, preferred_component


def _project_points_to_pixels(points, source):
    rows, cols, valid, _ = project_pixels(points, source)
    return rows, cols, valid


def _projection_indices(points, source):
    rows, cols, valid = _project_points_to_pixels(points, source)
    return np.where(valid, rows * source.image.shape[1] + cols, -1)


def _reserve_last_support(batch, projections, targets):
    """Keep at least one supporter for every covered original pixel in a batch."""
    for ids, target in zip(projections, targets):
        total = np.bincount(ids[ids >= 0], minlength=target.size)
        batch_ids = ids[batch]
        _, first, counts = np.unique(batch_ids, return_index=True, return_counts=True)
        pixels = batch_ids[first]
        valid = pixels >= 0
        first, counts, pixels = first[valid], counts[valid], pixels[valid]
        preserve = target.ravel()[pixels] & (counts == total[pixels])
        batch = np.delete(batch, first[preserve])
    return batch


def fast_projection_prune(
    hull,
    voxel_centers,
    optimized_sources,
    original_sources=None,
    max_passes=4,
    max_remove_fraction_per_pass=0.12,
    min_face_neighbors=2,
    redundancy_threshold=3.0,
    cleanup_each_pass=True,
    verbose=True,
):
    """Prune boundary voxels while preserving original rays and face connectivity."""
    sources = optimized_sources if original_sources is None else original_sources
    targets = [src.image for src in sources]
    centers = voxel_centers.reshape(*hull.shape, 3)
    hull = hull.astype(bool).copy()
    labels, count = connected_components_3d(hull)
    cc_removed = 0
    if count > 1:
        start_count = int(hull.sum())
        hull = labels == preferred_component(labels, count, centers, sources)
        cc_removed = start_count - int(hull.sum())
    if verbose:
        print(f"[POSTPROCESS] initial component cleanup: removed={cc_removed}, num_components={count}")

    total_removed = 0
    for pass_number in range(max_passes):
        if not hull.any():
            break
        points = centers[hull]
        neighbors = ndimage.convolve(hull.astype(np.uint8), FACE_NEIGHBORS, mode="constant")[hull] - 1
        projections = [_projection_indices(points, src) for src in sources]
        score = (4 - np.minimum(neighbors, 4).astype(float)) * 1.25
        score += 1.5 * (neighbors < min_face_neighbors)
        removable = neighbors < 5
        for ids, target in zip(projections, targets):
            valid = ids >= 0
            counts = np.bincount(ids[valid], minlength=target.size)
            support = counts[ids[valid]]
            wanted = target.ravel()[ids[valid]]
            removable[valid] &= (support > 1) | ~wanted
            score[valid] += np.minimum(support - 1, 6) + np.where(wanted, -0.75, 2.25)
        candidates = np.flatnonzero(removable & (score >= redundancy_threshold))
        if not len(candidates):
            break
        limit = max(1, int(max_remove_fraction_per_pass * len(points)))
        order = np.argsort(-score[candidates], kind="stable")
        batch = _reserve_last_support(candidates[order[:limit]], projections, targets)
        occupied = np.flatnonzero(hull)
        trial = hull.copy()
        while len(batch):
            trial.ravel()[occupied[batch]] = False
            if connected_components_3d(trial)[1] == 1:
                break
            trial.ravel()[occupied[batch]] = True
            batch = batch[:len(batch) // 2]
        if not len(batch):
            break
        hull = trial
        total_removed += len(batch)
        if verbose:
            print(f"[POSTPROCESS] pass {pass_number + 1}: removed={len(batch)}, final_voxels={int(hull.sum())}")

    return hull, {
        "bulk_removed": int(total_removed),
        "cc_removed": cc_removed,
        "final_voxels": int(hull.sum()),
    }
