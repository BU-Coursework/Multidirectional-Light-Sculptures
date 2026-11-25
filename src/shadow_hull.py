import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label
from scipy.spatial import cKDTree
from skimage.graph import MCP

from distances import silhouette_weights
from projections import pixel_indices, projected_mask


FACE_NEIGHBORS = generate_binary_structure(3, 1)


def compute_raw_shadow_hull(sources, voxel_centers, projections=None):
    points = voxel_centers.reshape(-1, 3)
    if projections is None:
        projections = (pixel_indices(points, src) for src in sources)
    keep = np.ones(len(points), dtype=bool)
    for src, ids in zip(sources, projections):
        keep &= (ids >= 0) & src.image.ravel()[np.maximum(ids, 0)]
    return keep.reshape(voxel_centers.shape[:3])


def connected_components_3d(mask):
    return label(mask, structure=FACE_NEIGHBORS)


def preferred_component(labels, count, voxel_centers, sources):
    """Prefer a component that projects more of the original target detail."""
    occupied = labels > 0
    components = labels[occupied].astype(np.int64)
    sizes = np.bincount(components, minlength=count + 1)
    points = voxel_centers[occupied]
    scores = np.zeros(count + 1)
    for src in sources:
        ids = pixel_indices(points, src)
        valid = ids >= 0
        size = src.image.size
        hits = np.unique(components[valid] * size + ids[valid])
        regions, pixels = np.divmod(hits, size)
        target = src.image.ravel()
        weights = silhouette_weights(src.image).ravel()
        overlap = np.bincount(regions[target[pixels]], weights[pixels[target[pixels]]], minlength=count + 1)
        extra = np.bincount(regions[~target[pixels]], weights[pixels[~target[pixels]]], minlength=count + 1)
        scores += overlap / np.maximum(weights[target].sum() + extra, 1)
    return int(np.lexsort((sizes[1:], scores[1:]))[-1] + 1)


def _local_bridge(a, b, voxel_centers, sources, weights, covered, connected):
    """Find a face-connected route near a closest pair of component voxels."""
    margin = 8
    lo = np.maximum(np.minimum(a, b) - margin, 0)
    hi = np.minimum(np.maximum(a, b) + margin + 1, connected.shape)
    area = tuple(slice(low, high) for low, high in zip(lo, hi))
    points = voxel_centers[area].reshape(-1, 3)
    cost = np.ones(len(points), dtype=np.float32)
    for src, weight, coverage in zip(sources, weights, covered):
        ids = pixel_indices(points, src)
        pixels = np.maximum(ids, 0)
        unwanted = (ids >= 0) & ~coverage[pixels] & ~src.image.ravel()[pixels]
        cost += np.where(ids < 0, 6, np.where(unwanted, 6 * weight[pixels], 0))
    cost = cost.reshape(tuple(hi - lo))
    cost[connected[area]] = 0.1
    solver = MCP(cost, fully_connected=False)
    start, end = tuple(a - lo), tuple(b - lo)
    solver.find_costs([start], [end])
    return np.asarray(solver.traceback(end)) + lo


def connect_all_components_fast(
    hull, min_component_size=1, verbose=False, *, voxel_centers, sources,
    max_added_shadow_fraction=0.08, bridge_radius=0,
):
    hull = hull.copy()
    labels, count = connected_components_3d(hull)
    if count <= 1:
        return hull
    main = preferred_component(labels, count, voxel_centers, sources)
    connected = labels == main
    main_points = np.argwhere(connected)
    tree = cKDTree(main_points)
    weights = [silhouette_weights(src.image).ravel() for src in sources]
    covered = [
        projected_mask(pixel_indices(voxel_centers[connected], src), src.image.shape).ravel()
        for src in sources
    ]
    budgets = [max(2, int(src.image.sum() * max_added_shadow_fraction)) for src in sources]
    sizes = np.bincount(labels.ravel())
    boxes = find_objects(labels)
    components = np.argsort(-sizes[1:], kind="stable") + 1
    attached = {main}
    neighbor_offsets = np.argwhere(FACE_NEIGHBORS) - 1

    def component_points(component):
        box = boxes[component - 1]
        offset = np.array([axis.start for axis in box])
        return np.argwhere(labels[box] == component) + offset

    for component in components:
        if component in attached or sizes[component] < min_component_size:
            continue
        points = component_points(component)
        distance, nearest = tree.query(points)
        i = np.argmin(distance)
        path = _local_bridge(main_points[nearest[i]], points[i], voxel_centers,
                             sources, weights, covered, connected)
        if bridge_radius:
            offsets = np.argwhere(generate_binary_structure(3, 1)) - 1
            expanded = path
            for _ in range(bridge_radius):
                expanded = np.unique((expanded[:, None] + offsets).reshape(-1, 3), axis=0)
            inside = ((expanded >= 0) & (expanded < hull.shape)).all(axis=1)
            path = expanded[inside]
        # A bridge can also connect a neighboring component along its route.
        neighbors = np.clip(path[:, None, :] + neighbor_offsets, 0, np.array(hull.shape) - 1)
        touched = set(np.unique(labels[tuple(neighbors.reshape(-1, 3).T)])) - attached - {0, component}
        attachment = np.vstack((path, points, *(component_points(k) for k in sorted(touched))))

        # Count distinct new background pixels from the entire attachment.
        added_points = voxel_centers[tuple(attachment.T)]
        added = [np.unique(ids[ids >= 0]) for ids in (pixel_indices(added_points, src) for src in sources)]
        new_extra = [
            int((~coverage[pixels] & ~src.image.ravel()[pixels]).sum())
            for coverage, pixels, src in zip(covered, added, sources)
        ]
        if any(extra > budget for extra, budget in zip(new_extra, budgets)):
            continue
        hull[tuple(path.T)] = True
        connected[tuple(attachment.T)] = True
        attached.update(touched | {component})
        for i, pixels in enumerate(added):
            covered[i][pixels] = True
            budgets[i] -= new_extra[i]
        if verbose:
            print(f"[SHADOW HULL] connected component {component}, new extra pixels per view: {new_extra}")
    return hull


def compute_shadow_hull(
    sources, voxel_centers, enforce_connectivity=True, min_component_size=1,
    verbose=False, *, original_sources=None, projections=None, bridge_radius=0,
):
    hull = compute_raw_shadow_hull(sources, voxel_centers, projections)
    if not enforce_connectivity:
        return hull
    return connect_all_components_fast(
        hull, min_component_size=min_component_size, verbose=verbose, voxel_centers=voxel_centers,
        sources=sources if original_sources is None else original_sources,
        bridge_radius=bridge_radius,
    )
