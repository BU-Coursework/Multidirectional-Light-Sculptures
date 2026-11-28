import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP

from export_mesh import mesh_from_voxels, mesh_triangles_in_world
from postprocess_prune import _reserve_last_support
from projections import pixel_indices, projected_mask
from render import render_mesh_shadow
from shadow_hull import connected_components_3d


def thin_voxels(hull):
    """Flag occupied voxels that belong to no solid 2×2×2 block."""
    return hull & ~ndi.binary_opening(hull, structure=np.ones((2, 2, 2), dtype=bool))


def _restore_connections(original, removed):
    kept = original.copy()
    kept.flat[removed] = False
    labels, count = connected_components_3d(kept)
    if count <= 1:
        return kept
    # Route only through occupied voxels of the original connected hull.
    sizes = np.bincount(labels.ravel())[1:]
    occupied = np.flatnonzero(kept)
    _, first = np.unique(labels.ravel()[occupied], return_index=True)
    anchors = [tuple(point) for point in np.array(np.unravel_index(occupied[first], original.shape)).T]
    start = anchors.pop(int(np.argmax(sizes)))
    cost = np.full(original.shape, np.inf)
    cost[original] = 1
    cost[kept] = .001
    routing = MCP(cost, fully_connected=False)
    routing.find_costs([start], anchors)
    for anchor in anchors:
        route = np.asarray(routing.traceback(anchor))
        kept[tuple(route.T)] = True
    return kept


def mesh_shadows(mesh, sources, world_size, pitch):
    triangles = mesh_triangles_in_world(mesh, world_size, pitch)
    return [render_mesh_shadow(triangles, source) for source in sources]


def shadow_regressions(before, after, sources):
    """Find lost target coverage and newly shadowed background in each view."""
    return [(source.image & old & ~new) | (~source.image & ~old & new)
            for old, new, source in zip(before, after, sources)]


def trim_for_print(hull, centers, sources, mesh, world_size, pitch):
    """Trim a connected hull while preserving its correct STL shadow pixels."""
    thin = thin_voxels(hull)
    if not thin.any() or thin.sum() == hull.sum():
        return None
    labels, _ = connected_components_3d(thin)
    # Prefer eliminating small fragments when several voxels support the same pixel.
    occupied = np.flatnonzero(hull)
    candidates = np.flatnonzero(thin[hull])
    sizes = np.bincount(labels.ravel())
    order = np.argsort(sizes[labels.ravel()[occupied[candidates]]], kind="stable")
    points = centers[hull]
    projections = [pixel_indices(points, source) for source in sources]
    protected = [source.image & projected_mask(ids, source.image.shape)
                 for source, ids in zip(sources, projections)]
    removals = occupied[_reserve_last_support(candidates[order], projections, protected)]
    if not removals.size:
        return None
    baseline = mesh_shadows(mesh, sources, world_size, pitch)
    for attempt in range(4):
        trial = _restore_connections(hull, removals)
        if trial.sum() == hull.sum():
            return None
        trial_mesh = mesh_from_voxels(trial, pitch)
        shadows = mesh_shadows(trial_mesh, sources, world_size, pitch)
        regressions = shadow_regressions(baseline, shadows, sources)
        if not any(mask.any() for mask in regressions):
            after = int(thin_voxels(trial).sum())
            if after < thin.sum() and trial_mesh.is_watertight and trial_mesh.body_count == 1:
                return trial, trial_mesh, {
                    "removed_voxels": int(hull.sum() - trial.sum()),
                    "thin_voxels_before": int(thin.sum()),
                    "thin_voxels_after": after,
                    "removed_extra_pixels": [int((~source.image & old & ~new).sum())
                                             for source, old, new in zip(sources, baseline, shadows)],
                }
            return None

        # Restore material near affected rays and widen the search if needed.
        removals = np.flatnonzero(hull & ~trial)
        points = centers.reshape(-1, 3)[removals]
        restore = np.zeros(len(removals), dtype=bool)
        for source, mask in zip(sources, regressions):
            if not mask.any():
                continue
            area = ndi.binary_dilation(mask, iterations=attempt + 1) if attempt else mask
            ids = pixel_indices(points, source)
            restore |= (ids >= 0) & area.ravel()[np.maximum(ids, 0)]
        removals = removals[~restore]
    return None
