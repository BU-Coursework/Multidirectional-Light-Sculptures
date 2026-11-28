from dataclasses import dataclass, replace
from statistics import mean

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, maximum_filter

from distances import shadow_detail_weights
from export_mesh import mesh_from_voxels, mesh_triangles_in_world
from render import doubled_shadow_source, render_mesh_shadow
from shadow_hull import FACE_NEIGHBORS, connected_components_3d
from simulate import evaluate_view


@dataclass
class SculptureCandidate:
    name: str
    hull: np.ndarray
    bridged: np.ndarray
    targets: list
    edited: list
    pruning: dict
    adjusted_setup: bool = False


def footprint_sampled_masks(sources, grid_size, coverage):
    """Area-resample fine masks to propose geometry on the coarser voxel grid."""
    candidates = []
    for src in sources:
        fine = Image.fromarray(src.image.astype(np.uint8) * 255)
        coarse = np.asarray(fine.resize((grid_size, grid_size), Image.Resampling.BOX))
        candidates.append(replace(src, image=coarse >= 255 * coverage))
    return candidates


def connected_core_width(hull, pitch):
    """Estimate a connected core width from face-neighbor erosions, up to seven voxels."""
    width = pitch
    for layer in range(1, 4):
        hull = binary_erosion(hull, structure=FACE_NEIGHBORS)
        if connected_components_3d(hull)[1] != 1:
            break
        width = (2 * layer + 1) * pitch
    return float(width)


def measure_candidate(candidate, world_size, pitch, *, mesh=None, fine_sampling=True):
    if mesh is None:
        mesh = mesh_from_voxels(candidate.hull, pitch)
    triangles = mesh_triangles_in_world(mesh, world_size, pitch)
    scores = []
    for source in candidate.targets:
        target = source.image
        shadow = render_mesh_shadow(triangles, source)
        holes = binary_fill_holes(target, structure=np.ones((3, 3), dtype=bool)) & ~target
        thin = target & (distance_transform_edt(target) <= 1.5)
        result = evaluate_view(target, shadow)
        result.update(
            thin_recall=float((thin & shadow).sum() / thin.sum()) if thin.any() else None,
            hole_preservation=float((holes & ~shadow).sum() / holes.sum()) if holes.any() else None,
        )
        if fine_sampling:
            fine_source = doubled_shadow_source(source)
            fine_target = fine_source.image
            fine_shadow = render_mesh_shadow(triangles, fine_source, cell_centers=True)
            weights = np.repeat(np.repeat(shadow_detail_weights(target), 2, axis=0), 2, axis=1)
            result["area_iou"] = evaluate_view(fine_target, fine_shadow)["iou"]
            result["detail_extra"] = float(weights[fine_shadow & ~fine_target].sum() / max(1, fine_target.sum()))
        scores.append(result)
    average = mean(item["iou"] for item in scores)
    weakest = min(item["iou"] for item in scores)
    detail = [item[key] for item in scores for key in ("thin_recall", "hole_preservation")
              if item[key] is not None]
    score = .55 * average + .30 * weakest + .15 * (mean(detail) if detail else average)
    area = [item.get("area_iou", item["iou"]) for item in scores]
    if fine_sampling:
        score = .55 * score + .45 * (.65 * mean(area) + .35 * min(area))
        score -= .06 * mean(item["detail_extra"] for item in scores)
    levels = np.flatnonzero(candidate.hull.any(axis=(0, 2)))
    above_bed = candidate.hull[:, levels[0]:levels[-1] + 1, :]
    supported = maximum_filter(above_bed[:, :-1, :], size=(3, 1, 3))
    unsupported = int((above_bed[:, 1:, :] & ~supported).sum())
    if fine_sampling:
        score -= .015 * unsupported / candidate.hull.sum()
    metrics = {
        "name": candidate.name, "mean_iou": average, "worst_iou": weakest, "views": scores,
        "score": score, "mean_area_iou": mean(area), "worst_area_iou": min(area),
        "occupied_voxels": int(candidate.hull.sum()),
        "mesh_volume": float(abs(mesh.volume)),
        "connected_core_width": connected_core_width(candidate.hull, pitch),
        "unsupported_voxels_y_up": unsupported,
        "adjusted_setup": candidate.adjusted_setup,
        "pruning": candidate.pruning,
    }
    return metrics, mesh


def select_candidate(measured):
    """Protect mean coverage on both rasters before ranking detail and overhangs."""
    floor = measured[0]["mean_iou"] - .005
    area_floor = measured[0].get("mean_area_iou", measured[0]["mean_iou"]) - .005
    eligible = [i for i, item in enumerate(measured) if item["mean_iou"] >= floor
                and item.get("mean_area_iou", item["mean_iou"]) >= area_floor]
    if "mean_area_iou" in measured[0]:
        # Detail and overhang terms cannot outweigh a material loss on both rasters.
        eligible = [
            i for i in eligible
            if not any(
                measured[j]["mean_iou"] > measured[i]["mean_iou"] + .0005
                and measured[j]["mean_area_iou"] > measured[i]["mean_area_iou"] + .0005
                for j in eligible
            )
        ]
    return max(eligible, key=lambda i: (measured[i]["score"],
               -measured[i].get("unsupported_voxels_y_up", 0), -measured[i]["mesh_volume"]))


def tradeoff_options(measured, chosen):
    def objectives(item):
        return item["score"], item["connected_core_width"], -item["mesh_volume"]

    options = []
    for i, item in enumerate(measured):
        if i == chosen:
            continue
        point = objectives(item)
        dominated = any(
            all(a >= b - 1e-9 for a, b in zip(objectives(other), point))
            and any(a > b + 1e-9 for a, b in zip(objectives(other), point))
            for j, other in enumerate(measured) if i != j
        )
        if not dominated:
            options.append(i)
    return options
