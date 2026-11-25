import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_fill_holes

from shadow_hull import compute_shadow_hull, connected_components_3d, preferred_component
from render import render_shadow
from projections import project_pixels
from warp import smooth_displacement, warp_mask
from distances import outside_distance, silhouette_weights


def inconsistent_pixels(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return desired & (~actual)


def boundary_pixels(mask: np.ndarray):
    boundary = mask & (~binary_erosion(mask))
    ys, xs = np.where(boundary)
    return np.stack([xs, ys], axis=1) if len(xs) > 0 else np.zeros((0, 2), dtype=int)


def nearest_boundary_point_from_list(boundary_pts: np.ndarray, target_xy):
    if len(boundary_pts) == 0:
        return None
    diffs = boundary_pts - np.asarray(target_xy)[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    return boundary_pts[np.argmin(d2)]


def precompute_source_projection_data(sources, voxel_centers):
    # Project all voxel centers into every source once to build a lookup table
    pts = voxel_centers.reshape(-1, 3)
    proj_data = []

    for src in sources:
        pyi, pxi, valid, depth = project_pixels(pts, src)
        width = src.image.shape[1]
        indices = np.flatnonzero(valid)
        pixels = pyi[indices] * width + pxi[indices]
        order = np.lexsort((depth[indices], pixels))
        indices, pixels = indices[order], pixels[order]
        keys, starts = np.unique(pixels, return_index=True)
        ends = np.r_[starts[1:], len(indices)]
        ray_lookup = {(int(key % width), int(key // width)): indices[start:end]
                      for key, start, end in zip(keys, starts, ends)}

        proj_data.append({
            "px": pxi,
            "py": pyi,
            "valid": valid,
            "depth": depth,
            "ray_lookup": ray_lookup,
        })

    return pts, proj_data


def candidate_voxel_cost_fast(point_idx, source_index, sources, outside_cost_maps, proj_data):
    total = 0.0

    for k, src in enumerate(sources):
        if k == source_index:
            continue

        pd = proj_data[k]
        if not pd["valid"][point_idx]:
            total += 1e6
            continue

        x = int(np.clip(pd["px"][point_idx], 0, src.image.shape[1] - 1))
        y = int(np.clip(pd["py"][point_idx], 0, src.image.shape[0] - 1))
        total += outside_cost_maps[k][y, x]

    return total


def find_least_cost_voxel_for_inconsistent_pixel_fast(
    source_index,
    pixel_xy,
    sources,
    outside_cost_maps,
    proj_data,
    max_ray_samples=None,
):
    # For a missing pixel in source_index, find the least-cost voxel along that backprojected ray.
    pd = proj_data[source_index]
    idxs = pd["ray_lookup"].get(pixel_xy, None)

    if idxs is None or len(idxs) == 0:
        return None

    if max_ray_samples is not None and len(idxs) > max_ray_samples:
        # sample uniformly along the sorted depth list
        sample_ids = np.linspace(0, len(idxs) - 1, max_ray_samples).astype(int)
        idxs = idxs[sample_ids]

    best_idx = None
    best_cost = float("inf")

    for idx in idxs:
        cost = candidate_voxel_cost_fast(
            point_idx=idx,
            source_index=source_index,
            sources=sources,
            outside_cost_maps=outside_cost_maps,
            proj_data=proj_data
        )
        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    if best_idx is None:
        return None

    return best_idx, best_cost


def project_point_to_image_fast(point_idx, src, pd):
    if not pd["valid"][point_idx]:
        return None
    x = int(np.clip(pd["px"][point_idx], 0, src.image.shape[1] - 1))
    y = int(np.clip(pd["py"][point_idx], 0, src.image.shape[0] - 1))
    return np.array([x, y], dtype=int)


def build_displacement_constraints(
    sources,
    voxel_centers,
    sample_per_view=300,
    max_ray_samples=24,
    *,
    actuals=None,
    proj_data=None,
    required_sources=None,
):
    if actuals is None:
        hull = compute_shadow_hull(sources, voxel_centers)
        actuals = [render_shadow(hull, voxel_centers, s) for s in sources]
    targets = sources if required_sources is None else required_sources
    inconsistencies = [inconsistent_pixels(s.image, a) for s, a in zip(targets, actuals)]
    outside_cost_maps = [outside_distance(s.image) for s in sources]
    boundary_pts_list = [boundary_pixels(s.image) for s in sources]

    dx_list = [np.zeros_like(s.image, dtype=float) for s in sources]
    dy_list = [np.zeros_like(s.image, dtype=float) for s in sources]
    add_maps = [np.zeros_like(s.image, dtype=bool) for s in sources]

    if proj_data is None:
        _, proj_data = precompute_source_projection_data(sources, voxel_centers)

    total_missing = 0

    for j, inc in enumerate(inconsistencies):
        ys, xs = np.where(inc)
        total_missing += len(xs)

        if len(xs) == 0:
            continue

        candidate_ids = np.arange(len(xs))

        if len(candidate_ids) > sample_per_view:
            candidate_ids = np.random.choice(
                candidate_ids,
                size=sample_per_view,
                replace=False,
            )

        for count, t in enumerate(candidate_ids, start=1):
            px = int(xs[t])
            py = int(ys[t])

            result = find_least_cost_voxel_for_inconsistent_pixel_fast(
                source_index=j,
                pixel_xy=(px, py),
                sources=sources,
                outside_cost_maps=outside_cost_maps,
                proj_data=proj_data,
                max_ray_samples=max_ray_samples,
            )

            if result is None:
                continue

            point_idx, _ = result

            for k, src in enumerate(sources):
                if k == j:
                    continue

                q = project_point_to_image_fast(point_idx, src, proj_data[k])
                if q is None:
                    continue

                qx, qy = int(q[0]), int(q[1])

                # discrete growth suggestion
                add_maps[k][qy, qx] = True

                # smooth boundary pull
                b = nearest_boundary_point_from_list(boundary_pts_list[k], q)
                if b is not None:
                    bx, by = int(b[0]), int(b[1])
                    dx_list[k][by, bx] += (qx - bx)
                    dy_list[k][by, bx] += (qy - by)

    stats = {
        "total_missing": int(total_missing),
        "missing_per_view": [int(i.sum()) for i in inconsistencies],
        "growth_pixels_per_view": [int(a.sum()) for a in add_maps],
    }

    return dx_list, dy_list, add_maps, actuals, inconsistencies, stats


def optimize_silhouettes(
    sources,
    voxel_centers,
    iterations=8,
    alpha=0.20,
    sigma=5.0,
    sample_per_view=300,
    growth_radius=2,
    max_ray_samples=24,
    plateau_patience=2,
    fallback_growth_threshold=5,
    fallback_global_dilation=True,
    verbose=True
):
    """Propose limited mask growth and keep the best connected original shadow."""
    current_sources = list(sources)
    best_sources = list(sources)
    best_score = -1.0
    stall_count = 0
    proj_data = None
    weights = [silhouette_weights(src.image) for src in sources]
    cleanup_structure = np.ones((3, 3), dtype=bool)
    growth_structure = np.ones((2 * growth_radius + 1, 2 * growth_radius + 1), dtype=bool)
    holes = [binary_fill_holes(src.image, structure=cleanup_structure) & ~src.image
             for src in sources]
    distances = [outside_distance(src.image) for src in sources]
    budgets = [max(2, int(src.image.sum() * .06)) if src.image.any() else 0 for src in sources]

    for it in range(iterations + 1):
        hull = compute_shadow_hull(current_sources, voxel_centers, original_sources=sources)
        labels, count = connected_components_3d(hull)
        if count > 1:
            hull = labels == preferred_component(labels, count, voxel_centers, sources)
        actuals = [render_shadow(hull, voxel_centers, src) for src in sources]
        scores = [
            weight[src.image & actual].sum() / max(weight[src.image | actual].sum(), 1)
            for src, actual, weight in zip(sources, actuals, weights)
        ]
        score = .75 * float(np.mean(scores)) + .25 * min(scores)
        if score > best_score:
            best_score = score
            best_sources = list(current_sources)
            stall_count = 0
        else:
            stall_count += 1
        if verbose:
            print(f"[OPTIMIZER] candidate {it}/{iterations}: original weighted overlap={score:.4f}")

        if score == 1 or stall_count >= plateau_patience or it == iterations:
            break
        if proj_data is None:
            _, proj_data = precompute_source_projection_data(sources, voxel_centers)

        dx_list, dy_list, add_maps, _, _, stats = build_displacement_constraints(
            current_sources, voxel_centers, sample_per_view=sample_per_view,
            max_ray_samples=max_ray_samples, actuals=actuals, proj_data=proj_data,
            required_sources=sources,
        )
        if not stats["total_missing"]:
            break
        worst_view = int(np.argmax(stats["missing_per_view"]))
        updated_sources = []
        for idx, (src, dx, dy, add_map) in enumerate(zip(current_sources, dx_list, dy_list, add_maps)):
            dx_s, dy_s = smooth_displacement(dx, dy, sigma=sigma)
            warped = warp_mask(src.image, dx=alpha * dx_s, dy=alpha * dy_s)
            grown = binary_dilation(add_map, structure=growth_structure)
            proposal = src.image | warped | grown
            if idx == worst_view and (stats["growth_pixels_per_view"][idx] < fallback_growth_threshold
                                      or (fallback_global_dilation and stall_count)):
                proposal = binary_dilation(proposal, structure=cleanup_structure)
            proposal |= binary_closing(proposal, structure=cleanup_structure)
            allowed = proposal & ~src.image & ~holes[idx]
            remaining = max(0, budgets[idx] - int((src.image & ~sources[idx].image).sum()))
            candidates = np.flatnonzero(allowed)
            # Prioritize actual missing-ray proposals over neighboring pixels.
            priority = np.lexsort((distances[idx].ravel()[candidates], ~add_map.ravel()[candidates]))
            new_img = src.image.copy()
            new_img.flat[candidates[priority[:remaining]]] = True
            updated_sources.append(type(src)(
                image=new_img, direction=src.direction, up=src.up,
                world_center=src.world_center, world_size=src.world_size,
            ))
        if all(np.array_equal(old.image, new.image) for old, new in zip(current_sources, updated_sources)):
            break
        current_sources = updated_sources

    if verbose:
        print(f"[OPTIMIZER] best original weighted overlap: {best_score:.4f}")
    return best_sources
