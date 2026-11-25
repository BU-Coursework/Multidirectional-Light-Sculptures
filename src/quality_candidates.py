from dataclasses import dataclass, replace
from statistics import mean

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt

from export_mesh import mesh_from_voxels, mesh_triangles_in_world
from render import render_mesh_shadow
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


def measure_candidate(candidate, world_size, pitch):
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
        scores.append(result)
    average = mean(item["iou"] for item in scores)
    weakest = min(item["iou"] for item in scores)
    detail = [item[key] for item in scores for key in ("thin_recall", "hole_preservation")
              if item[key] is not None]
    metrics = {
        "name": candidate.name, "mean_iou": average, "worst_iou": weakest, "views": scores,
        "score": .55 * average + .30 * weakest + .15 * (mean(detail) if detail else average),
        "occupied_voxels": int(candidate.hull.sum()),
        "mesh_volume": float(abs(mesh.volume)),
        "connected_core_width": connected_core_width(candidate.hull, pitch),
        "adjusted_setup": candidate.adjusted_setup,
        "pruning": candidate.pruning,
    }
    return metrics, mesh


def select_candidate(measured):
    """Favor mesh fidelity and keep mean IoU within half a point of the incumbent."""
    floor = measured[0]["mean_iou"] - .005
    eligible = [i for i, item in enumerate(measured) if item["mean_iou"] >= floor]
    return max(eligible, key=lambda i: (measured[i]["score"], -measured[i]["mesh_volume"]))


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
