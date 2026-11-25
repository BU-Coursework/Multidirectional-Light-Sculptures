from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from projections import normalize, pixel_indices, projected_mask
from shadow_hull import compute_raw_shadow_hull
from voxel_ops import make_voxel_centers


@dataclass(frozen=True)
class ViewAdjustment:
    horizontal: float = 0.0
    vertical: float = 0.0
    scale: float = 1.0
    yaw: float = 0.0
    pitch: float = 0.0

    def apply(self, source):
        right = normalize(np.cross(source.up, source.direction))
        up = normalize(np.cross(source.direction, right))
        yaw_rotation = Rotation.from_rotvec(np.deg2rad(self.yaw) * up)
        pitch_rotation = Rotation.from_rotvec(np.deg2rad(self.pitch) * right)
        rotation = pitch_rotation * yaw_rotation
        center = source.world_center + source.world_size * rotation.apply(
            self.horizontal * right + self.vertical * up)
        return replace(source, direction=rotation.apply(source.direction), up=rotation.apply(source.up),
                       world_center=center, world_size=source.world_size * self.scale)


def search_light_setup(sources, grid, *, alignment, angles):
    """Find a bounded setup change using strict coverage on a coarse grid."""
    size = min(grid, 36)
    centers = make_voxel_centers(size, size, size, sources[0].world_size)
    points = centers.reshape(-1, 3)
    coarse = [
        replace(src, image=np.asarray(Image.fromarray(src.image).resize((size, size), Image.Resampling.NEAREST)))
        for src in sources
    ]
    adjustments = [ViewAdjustment() for _ in sources]
    projections = [pixel_indices(points, src) for src in coarse]

    def score(indices):
        raw = compute_raw_shadow_hull(coarse, centers, indices).ravel()
        coverage = [
            (projected_mask(ids[raw], src.image.shape) & src.image).sum() / max(1, src.image.sum())
            for src, ids in zip(coarse, indices)
        ]
        return .7 * float(np.mean(coverage)) + .3 * min(coverage)

    baseline = best = score(projections)
    alignment_steps = {
        "horizontal": (-.12, -.09, -.06, -.03, .03, .06, .09, .12),
        "vertical": (-.12, -.09, -.06, -.03, .03, .06, .09, .12),
        "scale": (.90, .95, 1.05, 1.10),
    }
    angle_steps = {"yaw": (-12, -6, 6, 12), "pitch": (-12, -6, 6, 12)}
    for enabled, steps in ((alignment, alignment_steps), (angles, angle_steps)):
        if not enabled:
            continue
        for view in range(1, len(sources)):
            for field, values in steps.items():
                current = adjustments[view]
                winning_ids = projections[view]
                for value in values:
                    candidate = replace(current, **{field: value})
                    ids = pixel_indices(points, candidate.apply(coarse[view]))
                    trial = list(projections)
                    trial[view] = ids
                    candidate_score = score(trial)
                    if candidate_score > best + 1e-6:
                        best = candidate_score
                        adjustments[view] = candidate
                        winning_ids = ids
                projections[view] = winning_ids
    return [adjustment.apply(src) for adjustment, src in zip(adjustments, sources)], adjustments, baseline, best
