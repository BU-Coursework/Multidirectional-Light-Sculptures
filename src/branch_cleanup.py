"""Remove terminal thin material and fill small vacancies under STL shadow limits."""

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi

from distances import outside_distance, shadow_detail_weights, silhouette_weights
from print_cleanup import mesh_shadows, thin_voxels
from projections import pixel_indices
from quality_candidates import measure_candidate
from shadow_hull import FACE_NEIGHBORS, connected_components_3d


MAX_BRANCH_SIZE = 48
MAX_FILL_SIZE = 20
BATCH_FRACTIONS = (.12, .34, .62, 1.)
DETAIL_WINDOW = 15
MAX_DETAIL_ERRORS = 4
NEIGHBOR_KERNEL = FACE_NEIGHBORS.astype(np.uint8)
NEIGHBOR_KERNEL[1, 1, 1] = 0


def _component_boxes(labels):
    for label, box in enumerate(ndi.find_objects(labels), 1):
        padded = tuple(slice(max(0, sl.start - 1), min(length, sl.stop + 1))
                       for sl, length in zip(box, labels.shape))
        yield label, padded, labels[padded] == label


def _terminal_branches(hull, thin):
    labels, _ = connected_components_3d(thin)
    cores, _ = connected_components_3d(hull & ~thin)
    sizes = np.bincount(labels.ravel())
    branches = []
    for label, box, part in _component_boxes(labels):
        if sizes[label] > MAX_BRANCH_SIZE:
            continue
        touches = ndi.binary_dilation(part, structure=FACE_NEIGHBORS) & (cores[box] > 0)
        if np.unique(cores[box][touches]).size != 1:
            continue
        collar = ndi.binary_dilation(touches, structure=np.ones((3, 3, 3), dtype=bool))
        if connected_components_3d(collar)[1] == 1:
            origin = [sl.start for sl in box]
            branches.append(np.ravel_multi_index((np.argwhere(part) + origin).T, hull.shape))
    return branches


def _crowded_vacancies(hull):
    neighbors = ndi.convolve(hull.astype(np.uint8), NEIGHBOR_KERNEL, mode="constant")
    return ~hull & (neighbors >= 4)


def _small_fills(vacancies):
    labels, _ = connected_components_3d(vacancies)
    fills = []
    for label, box in enumerate(ndi.find_objects(labels), 1):
        points = np.argwhere(labels[box] == label) + [sl.start for sl in box]
        if len(points) <= MAX_FILL_SIZE:
            fills.append(np.ravel_multi_index(points.T, vacancies.shape))
    return fills


def _core_fraction(hull):
    eroded = ndi.binary_erosion(hull, structure=FACE_NEIGHBORS)
    labels, count = connected_components_3d(eroded)
    if count == 0:
        return 0.
    sizes = np.bincount(labels[eroded])
    return float(sizes.max() / eroded.sum())


def _weak_separations(hull, minimum_size):
    """Count thin components whose removal separates substantial regions."""
    thin = thin_voxels(hull)
    labels, _ = connected_components_3d(thin)
    cores, _ = connected_components_3d(hull & ~thin)
    core_sizes = np.bincount(cores.ravel())
    separations = []
    for _, box, part in _component_boxes(labels):
        border = ndi.binary_dilation(part, structure=FACE_NEIGHBORS) & ~part
        touched = np.unique(cores[box][border])
        if touched.size < 2 or np.count_nonzero(core_sizes[touched[touched > 0]] >= 64) < 2:
            continue
        remaining = hull.copy()
        remaining[box] &= ~part
        components, count = connected_components_3d(remaining)
        if count > 1:
            sizes = np.bincount(components.ravel())[1:]
            second = int(np.partition(sizes, -2)[-2])
            if second >= minimum_size:
                separations.append(second)
    return len(separations), max(separations, default=0)


def _keeps_weak_separations(before, after):
    return after[0] <= before[0] and after[1] <= 1.05 * before[1]


@dataclass
class _Geometry:
    thin: int
    branches: list
    vacancies: np.ndarray
    core_fraction: float

    @classmethod
    def measure(cls, hull):
        thin = thin_voxels(hull)
        return cls(int(thin.sum()), _terminal_branches(hull, thin), _crowded_vacancies(hull),
                   _core_fraction(hull))

    def counts(self):
        return self.thin, sum(map(len, self.branches)), int(self.vacancies.sum())


def preserves_branch_cleanup(before, after):
    """Check that a later hollowing pass keeps the cleanup's structural gains."""
    if connected_components_3d(after)[1] != 1:
        return False
    base, carved = _Geometry.measure(before), _Geometry.measure(after)
    minimum = max(500, int(before.sum() * .01))
    return (all(a <= b for a, b in zip(carved.counts(), base.counts()))
            and carved.core_fraction >= base.core_fraction - .005
            and _keeps_weak_separations(_weak_separations(before, minimum),
                                        _weak_separations(after, minimum)))


def _rank_edits(hull, groups, centers, sources, shadows, weights, near_edges, *, fill):
    if not groups:
        return []
    occupied = centers[hull]
    points = centers.reshape(-1, 3)[np.concatenate(groups)]
    splits = np.cumsum([len(group) for group in groups])[:-1]
    costs = np.zeros((len(groups), len(sources)))
    eligible = np.ones(len(groups), dtype=bool)
    for view, source in enumerate(sources):
        target = source.image.ravel()
        ids = pixel_indices(occupied, source)
        support = np.bincount(ids[ids >= 0], minlength=target.size)
        for index, proposed in enumerate(np.split(pixel_indices(points, source), splits)):
            pixels, counts = np.unique(proposed[proposed >= 0], return_counts=True)
            if fill:
                covered = shadows[view].ravel()[pixels] | (support[pixels] > 0)
                if len(pixels) == 0 or (proposed < 0).any() or not (covered | near_edges[view][pixels]).all():
                    eligible[index] = False
                changed = ~covered & ~target[pixels]
            else:
                changed = (support[pixels] == counts) & target[pixels]
            costs[index, view] = weights[view][pixels[changed]].sum() / len(groups[index])
    return [groups[i] for i in sorted(np.flatnonzero(eligible),
            key=lambda i: (costs[i].max(), costs[i].sum(), -len(groups[i])))]


@dataclass
class _Trial:
    hull: np.ndarray
    geometry: _Geometry
    metrics: dict
    mesh: object
    shadows: list
    operations: tuple = ()
    gain: float = 0.

    def rank(self):
        return self.gain, self.metrics["mean_area_iou"]


def _blur(image):
    return ndi.gaussian_filter(image.astype(float), .85)


def _too_many_local_errors(errors):
    # Limit concentrated damage to a feature while allowing sparse changes
    # in unrelated parts of the same shadow.
    density = ndi.uniform_filter(errors.astype(float), size=DETAIL_WINDOW, mode="constant")
    return float(density.max()) * DETAIL_WINDOW ** 2 > MAX_DETAIL_ERRORS + .5


def _within_shadow_budget(before, after, targets, protected, blurred_targets):
    for old, new, view, old_shadow, new_shadow, detail, blurred in zip(
        before.metrics["views"], after.metrics["views"], targets, before.shadows, after.shadows,
        protected, blurred_targets,
    ):
        if new["area_iou"] < old["area_iou"] - .01:
            return False
        if old["hole_preservation"] is not None and new["hole_preservation"] < old["hole_preservation"] - .02:
            return False
        if _too_many_local_errors(detail & (old_shadow == view) & (new_shadow != view)):
            return False
        old_error = np.mean(np.abs(blurred - _blur(old_shadow)))
        if np.mean(np.abs(blurred - _blur(new_shadow))) > old_error + .002:
            return False
    return True


def clean_branches(candidate, centers, mesh, metrics, world_size, pitch):
    """Try small groups of edits to a selected hull and return a vetted candidate."""
    if connected_components_3d(candidate.hull)[1] != 1:
        return None
    sources = candidate.targets
    targets = [source.image for source in sources]
    weights = [np.maximum(silhouette_weights(target), shadow_detail_weights(target)) for target in targets]
    protected = [weight > 1 for weight in weights]
    weights = [weight.ravel() for weight in weights]
    near_edges = [((outside_distance(target) <= 1.42) & ~detail).ravel()
                  for target, detail in zip(targets, protected)]
    blurred_targets = [_blur(target) for target in targets]
    base = _Trial(candidate.hull, _Geometry.measure(candidate.hull), metrics, mesh,
                  mesh_shadows(mesh, sources, world_size, pitch))
    baseline_counts = base.geometry.counts()
    eligible = []

    def try_batch(hull, operations):
        geometry = _Geometry.measure(hull)
        counts = geometry.counts()
        if (counts[0] > baseline_counts[0] or counts[2] > baseline_counts[2]
                or geometry.core_fraction < base.geometry.core_fraction - .005):
            return None
        gain = float(sum(weight * (old - new) / max(1, old) for weight, old, new in zip(
            (.55, .25, .20), baseline_counts, counts,
        )))
        if gain <= 0 or connected_components_3d(hull)[1] != 1:
            return None
        proposal = replace(candidate, name="Branch trim cleanup", hull=hull)
        trial_metrics, trial_mesh = measure_candidate(proposal, world_size, pitch)
        if (not trial_mesh.is_watertight or trial_mesh.body_count != 1
                or trial_metrics["connected_core_width"] < metrics["connected_core_width"]):
            return None
        trial = _Trial(hull, geometry, trial_metrics, trial_mesh,
                       mesh_shadows(trial_mesh, sources, world_size, pitch), operations, gain)
        if _within_shadow_budget(base, trial, targets, protected, blurred_targets):
            eligible.append(trial)
            return trial
        return None

    def stage(current, *, fill):
        groups = _small_fills(current.geometry.vacancies) if fill else current.geometry.branches
        ranked = _rank_edits(current.hull, groups, centers, sources, current.shadows,
                             weights, near_edges, fill=fill)
        if not ranked:
            return current
        batches = sorted({max(1, int(fraction * len(ranked))) for fraction in BATCH_FRACTIONS})
        options = [current]
        for count in batches:
            hull = current.hull.copy()
            hull.flat[np.concatenate(ranked[:count])] = fill
            operation = "fill" if fill else "trim"
            trial = try_batch(hull, current.operations + (operation,))
            if trial is not None:
                options.append(trial)
        return max(options, key=_Trial.rank)

    trimmed = stage(base, fill=False)
    filled = stage(base, fill=True)
    # A trim can expose a different set of local fills and terminal branches.
    start = stage(trimmed, fill=True) if trimmed is not base else filled
    if start is not base:
        stage(start, fill=False)
    if not eligible:
        return None
    minimum = max(500, int(base.hull.sum() * .01))
    before = _weak_separations(base.hull, minimum)
    for trial in sorted(eligible, key=_Trial.rank, reverse=True):
        after = _weak_separations(trial.hull, minimum)
        if not _keeps_weak_separations(before, after):
            continue
        stats = {
            "operations": list(trial.operations),
            "removed_voxels": int((base.hull & ~trial.hull).sum()),
            "added_voxels": int((~base.hull & trial.hull).sum()),
            "thin_voxels_before": base.geometry.thin,
            "thin_voxels_after": trial.geometry.thin,
            "terminal_groups_before": len(base.geometry.branches),
            "terminal_groups_after": len(trial.geometry.branches),
            "crowded_vacancies_before": baseline_counts[2],
            "crowded_vacancies_after": int(trial.geometry.vacancies.sum()),
            "major_weak_separations_before": before[0],
            "major_weak_separations_after": after[0],
            "area_iou_deltas": [new["area_iou"] - old["area_iou"] for old, new in
                                zip(base.metrics["views"], trial.metrics["views"])],
        }
        return replace(candidate, name=trial.metrics["name"], hull=trial.hull), trial.metrics, trial.mesh, stats
    return None
