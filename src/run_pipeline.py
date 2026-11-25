import argparse
from dataclasses import asdict
from itertools import combinations
import numpy as np
import time

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_raw_shadow_hull, compute_shadow_hull, connected_components_3d
from shadow_source import build_sources
from export_mesh import mesh_triangles_in_world
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from projections import pixel_indices, projected_mask
from render import render_shadow
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from postprocess_prune import fast_projection_prune
from light_setup import search_light_setup
from quality_candidates import (
    SculptureCandidate, footprint_sampled_masks, measure_candidate, select_candidate, tradeoff_options,
)
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore", module="paramiko")

def print_view_metrics(name, summaries):
    print(f"\n{name}")
    ious = []
    for i, m in enumerate(summaries):
        ious.append(m["iou"])
        print(
            f"  view {i}: "
            f"IoU={m['iou']:.4f}, "
            f"target={m['target_pixels']}, "
            f"actual={m['actual_pixels']}, "
            f"missing={m['missing_pixels']}, "
            f"extra={m['extra_pixels']}"
        )
    if len(ious) > 0:
        print(f"  avg IoU = {float(np.mean(ious)):.4f}")
        print(f"  min IoU = {float(np.min(ious)):.4f}")


def parse_direction_string(s: str):
    """
    Example:
      "1,0,0;0,0,1"
    """
    dirs = []
    parts = s.strip().split(";")
    for p in parts:
        vals = [float(x) for x in p.split(",")]
        if len(vals) != 3:
            raise ValueError("Each direction must have 3 values.")
        dirs.append(np.array(vals, dtype=float))
    return dirs

def make_run_output_dir(base_dir="outputs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(base_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a shadow sculpture from one to three silhouette images."
    )

    parser.add_argument(
        "views",
        nargs="+",
        help="One to three silhouette image paths"
    )

    parser.add_argument(
        "--world-size",
        type=float,
        default=80.0,
        help="Physical size of the voxel world in mm"
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=160,
        help="Voxel resolution used for nx=ny=nz"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square size used to resize silhouette images (defaults to --grid)"
    )

    parser.add_argument(
        "--directions",
        type=str,
        default=None,
        help='Semicolon-separated direction vectors, e.g. "1,0,0;0,0,1"'
    )

    parser.add_argument(
        "--optimize-material",
        action="store_true",
        help="Enable hollow-shell carving"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binarization threshold for input image"
    )

    parser.add_argument(
        "--close-iters",
        type=int,
        default=0,
        help="Binary closing iterations for silhouette cleanup"
    )

    parser.add_argument(
        "--open-iters",
        type=int,
        default=0,
        help="Binary opening iterations for silhouette cleanup"
    )

    parser.add_argument(
        "--dilate-iters",
        type=int,
        default=0,
        help="Optional binary dilation iterations"
    )

    parser.add_argument(
        "--opt-iters",
        type=int,
        default=6,
        help="Silhouette optimization iterations"
    )
    parser.add_argument("--center-ranking", action="store_true",
                        help="Use the voxel-center optimizer choice without comparing STL candidates")
    parser.add_argument("--search-alignment", action="store_true",
                        help="Try bounded offsets and scale changes for views after the first")
    parser.add_argument("--search-light-angles", action="store_true",
                        help="Try bounded light angle changes for views after the first")
    parser.add_argument("--bridge-diameter-voxels", type=int, choices=(1, 3, 5), default=3,
                        help="Diameter of the optional thicker-bridge candidate")

    args = parser.parse_args()

    if len(args.views) > 3:
        parser.error("Please provide one to three silhouette images.")
    if args.center_ranking and (args.search_alignment or args.search_light_angles):
        parser.error("Setup search requires STL candidate ranking.")

    return args


def assess_strict_hull(sources, voxel_centers):
    points = voxel_centers.reshape(-1, 3)
    projections = [pixel_indices(points, src) for src in sources]
    raw = compute_raw_shadow_hull(sources, voxel_centers, projections)
    views = []
    for src, ids in zip(sources, projections):
        reachable = projected_mask(ids, src.image.shape)
        strict = projected_mask(ids[raw.ravel()], src.image.shape)
        views.append({
            "target_pixels": int(src.image.sum()),
            "no_center_ray": int((src.image & ~reachable).sum()),
            "blocked_by_other_views": int((src.image & reachable & ~strict).sum()),
            "strict_coverage": int((src.image & strict).sum()),
        })
    disjoint_view_pairs = []
    if len(sources) > 1 and not raw.any():
        valid_masks = [
            (ids >= 0) & src.image.ravel()[np.maximum(ids, 0)]
            for src, ids in zip(sources, projections)
        ]
        disjoint_view_pairs = [
            [i + 1, j + 1] for i, j in combinations(range(len(sources)), 2)
            if not (valid_masks[i] & valid_masks[j]).any()
        ]
    return raw, {
        "strict_voxels": int(raw.sum()), "disjoint_view_pairs": disjoint_view_pairs, "views": views,
    }


def _setup_description(sources, adjustments=None):
    views = [
        {"direction": src.direction.tolist(), "up": src.up.tolist(),
         "world_center": src.world_center.tolist(), "frame_size": src.world_size}
        for src in sources
    ]
    if adjustments is not None:
        for view, adjustment in zip(views, adjustments):
            view.update(asdict(adjustment))
    return views


def run_pipeline(
    view_paths,
    world_size=1.0,
    grid=120,
    image_size_value=None,
    optimize_material=False,
    output_dir=None,
    prune_passes=6,
    log=print,
    directions=None,
    opt_iters=6,
    threshold=128,
    close_iters=0,
    open_iters=0,
    dilate_iters=0,
    rank_stl=True,
    search_alignment=False,
    search_light_angles=False,
    bridge_diameter_voxels=3,
):
    t0 = time.time()

    if output_dir is None:
        output_dir = make_run_output_dir()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug"
    sim_dir = output_dir / "sim"
    mesh_dir = output_dir / "meshes"

    (debug_dir / "masks" / "base").mkdir(parents=True, exist_ok=True)
    (debug_dir / "masks" / "opt").mkdir(parents=True, exist_ok=True)
    (debug_dir / "slices").mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    log(f"[PIPELINE] output directory: {output_dir}")

    nx = ny = nz = grid
    image_size_value = grid if image_size_value is None else image_size_value
    image_size = (image_size_value, image_size_value)

    images = []
    for i, path in enumerate(view_paths):
        img = load_binary_image(path, size=image_size, threshold=threshold,
                                close_iters=close_iters, open_iters=open_iters, dilate_iters=dilate_iters)
        images.append(img)

    for i, img in enumerate(images):
        save_mask(img, str(debug_dir / "masks" / "base" / f"view{i}_mask.png"))

    sources = build_sources(images, world_size, directions=directions)
    original_sources = sources
    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)
    pitch = voxel_pitch(world_size, nx, ny, nz)
    strict, fit = assess_strict_hull(original_sources, voxel_centers)
    for i, view in enumerate(fit["views"]):
        log(f"[FIT] view {i + 1}: {view['no_center_ray']} pixels without a voxel-center ray, "
            f"{view['blocked_by_other_views']} ruled out by other silhouettes")
    for first, second in fit["disjoint_view_pairs"]:
        log(f"[FIT] views {first} and {second} share no strict voxel volume. "
            "Check mask placement and light directions if the edited result loses detail.")

    searched_sources = None
    setup_search = None
    if rank_stl and len(sources) > 1 and (search_alignment or search_light_angles):
        adjusted, adjustments, baseline, best = search_light_setup(
            original_sources, grid, alignment=search_alignment, angles=search_light_angles)
        setup_search = {"coarse_before": baseline, "coarse_after": best,
                        "views": _setup_description(adjusted, adjustments)}
        if best > baseline:
            searched_sources = adjusted
        log(f"[FIT] coarse setup search: {baseline:.1%} to {best:.1%}")

    history = []
    if len(sources) > 1:
        log("[PIPELINE] optimizing silhouettes...")

        optimized_sources = optimize_silhouettes(
            sources,
            voxel_centers,
            iterations=opt_iters,
            alpha=0.15,
            sigma=4.0,
            sample_per_view=300,
            growth_radius=2,
            verbose=True,
            on_candidate=(lambda score, views: history.append((score, views))) if rank_stl else None,
        )
    else:
        log("[PIPELINE] skipping optimization (single view)")
        optimized_sources = sources

    def build(name, edited, targets, passes, bridge_radius=0, bridged=None):
        if bridged is None:
            bridged = compute_shadow_hull(edited, voxel_centers, original_sources=targets,
                                          bridge_radius=bridge_radius)
        voxels, stats = fast_projection_prune(
            bridged, voxel_centers, edited, original_sources=targets, max_passes=passes,
            max_remove_fraction_per_pass=0.15, min_face_neighbors=2,
            redundancy_threshold=2.0, verbose=False,
        )
        return SculptureCandidate(name, voxels, bridged, targets, edited, stats, targets is searched_sources)

    log("[PIPELINE] building sculpture candidates...")
    incumbent = build("Voxel-ranked hull", optimized_sources, original_sources, prune_passes)
    candidates = [incumbent]
    if rank_stl:
        if prune_passes:
            candidates.append(build("Unpruned hull", optimized_sources, original_sources, 0,
                                    bridged=incumbent.bridged))
        if any(not np.array_equal(a.image, b.image) for a, b in zip(original_sources, optimized_sources)):
            candidates.append(build("Unedited targets", original_sources, original_sources, prune_passes))
        for _, proposal in sorted(history, key=lambda entry: entry[0], reverse=True):
            if (any(not np.array_equal(a.image, b.image) for a, b in zip(proposal, optimized_sources))
                    and any(not np.array_equal(a.image, b.image) for a, b in zip(proposal, original_sources))):
                candidates.append(build("Alternate edited targets", proposal, original_sources, prune_passes))
                break
        if bridge_diameter_voxels > 1 and connected_components_3d(strict)[1] > 1:
            candidates.append(build("Wider bridges", optimized_sources, original_sources, 0,
                                    bridge_radius=bridge_diameter_voxels // 2))
        if searched_sources is not None:
            candidates.append(build("Adjusted light setup", searched_sources, searched_sources, prune_passes))
        if image_size_value > grid:
            for name, coverage in (("Area-sampled masks", .50), ("Detail-preserving area masks", .25)):
                sampled = footprint_sampled_masks(original_sources, grid, coverage)
                candidates.append(build(name, sampled, original_sources, prune_passes))

    measured, meshes, accepted = [], [], []
    for candidate in candidates:
        if not candidate.hull.any():
            continue
        if any(candidate.targets is previous.targets and np.array_equal(candidate.hull, previous.hull)
               for previous in accepted):
            continue
        metrics, mesh = measure_candidate(candidate, world_size, pitch)
        measured.append(metrics)
        meshes.append(mesh)
        accepted.append(candidate)
        log(f"[STL] {candidate.name}: {metrics['mean_iou']:.1%} mean IoU, "
            f"{metrics['worst_iou']:.1%} worst view")
    if not accepted:
        (output_dir / "fit_diagnostics.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
        raise ValueError("The silhouettes leave no shared sculpture volume within the allowed edits. "
                         "Check their alignment and light directions.")
    chosen = select_candidate(measured)
    selected = accepted[chosen]
    hull, connected_hull = selected.hull, selected.bridged
    optimized_sources, target_sources = selected.edited, selected.targets
    edited_comparison = optimized_sources if optimized_sources[0].image.shape == image_size else None
    selected_setup = _setup_description(target_sources)
    log(f"[STL] selected {selected.name}")
    if len(target_sources) > 1:
        for i, src in enumerate(optimized_sources):
            save_mask(src.image, str(debug_dir / "masks" / "opt" / f"view{i}_optimized_mask.png"))
    alternatives = set(tradeoff_options(measured, chosen))
    alternatives.update(i for i, option in enumerate(accepted) if option.adjusted_setup and i != chosen)
    for i in sorted(alternatives):
        alternative = accepted[i]
        path = mesh_dir / f"alternative_{i + 1}.stl"
        meshes[i].export(path)
        measured[i]["stl_path"] = str(path)
        measured[i]["setup"] = _setup_description(alternative.targets)
        alternative_sim = output_dir / "candidates" / f"alternative_{i + 1}"
        simulate_and_save(alternative.hull, voxel_centers, alternative.targets,
                          out_dir=str(alternative_sim), prefix="hull",
                          triangles=mesh_triangles_in_world(meshes[i], world_size, pitch))
        measured[i]["sim_dir"] = str(alternative_sim)
    quality = {"selected": chosen, "candidates": measured, "selected_setup": selected_setup,
               "setup_search": setup_search, "selected_adjusted_setup": selected.adjusted_setup}

    for i, src in enumerate(target_sources):
        strict_actual = render_shadow(strict, voxel_centers, original_sources[i])
        hull_actual = render_shadow(hull, voxel_centers, src)
        bridged_actual = render_shadow(connected_hull, voxel_centers, src)
        fit["views"][i].update(
            lost_from_strict_hull=int((src.image & strict_actual & ~hull_actual).sum()),
            lost_at_connectivity_cleanup=int((src.image & bridged_actual & ~hull_actual).sum()),
        )
    (output_dir / "fit_diagnostics.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    hull_stl_path = str(mesh_dir / "shadow_hull.stl")

    log("[PIPELINE] exporting STL...")
    mesh = meshes[chosen]
    mesh.export(hull_stl_path)
    log(f"[PIPELINE] saved raw hull mesh: {hull_stl_path}")
    hull_summaries = simulate_and_save(
        hull, voxel_centers, target_sources, out_dir=str(sim_dir), prefix="hull",
        triangles=mesh_triangles_in_world(mesh, world_size, pitch), edited_sources=edited_comparison,
    )

    carved_stl_path = None
    carved_summaries = None

    if optimize_material:
        log("[PIPELINE] carving material...")
        carved, _ = carve_hollow_shell_strict(
            hull,
            voxel_centers,
            target_sources,
            shell_thickness_voxels=2,
            max_passes=4,
            random_seed=0,
            protect_endcaps=True,
            cleanup_components=True,
            min_component_size=150,
            verbose=True,
        )

        carved_mesh = mesh
        carved_count = int(carved.sum())
        if connected_components_3d(carved)[1] != 1:
            log("[PIPELINE] hollowing would break face connectivity; keeping the solid sculpture")
            carved = hull
        elif carved_count < hull.sum():
            hollow = SculptureCandidate("Hollow hull", carved, connected_hull, target_sources,
                                        optimized_sources, selected.pruning, selected.adjusted_setup)
            hollow_metrics, hollow_mesh = measure_candidate(hollow, world_size, pitch)
            solid_metrics = measured[chosen]
            keep_hollow = (hollow_metrics["mean_iou"] >= solid_metrics["mean_iou"] - .005
                           and hollow_metrics["score"] >= solid_metrics["score"] - .005)
            quality["hollow"] = {"accepted": keep_hollow, "metrics": hollow_metrics}
            if keep_hollow:
                carved_mesh = hollow_mesh
            else:
                log("[PIPELINE] hollowing reduced STL shadow fidelity; keeping the solid sculpture")
                carved = hull
        carved_count = int(carved.sum())
        hull_count = int(hull.sum())
        log(f"[PIPELINE] carved sculpture voxels: {carved_count}")
        log(f"[PIPELINE] carved sculpture occupancy ratio: {float(carved.mean())}")
        log(f"[PIPELINE] carved sculpture voxels removed: {hull_count - carved_count}")
        log(f"[PIPELINE] carved sculpture reduction ratio: {1 - carved_count / hull_count}")

        save_voxel_slices(hull, str(debug_dir / "slices"), "hull")
        save_voxel_slices(carved, str(debug_dir / "slices"), "carved")

        carved_stl_path = str(mesh_dir / "shadow_carved.stl")
        carved_mesh.export(carved_stl_path)
        carved_summaries = simulate_and_save(
            carved, voxel_centers, target_sources, out_dir=str(sim_dir), prefix="carved",
            triangles=mesh_triangles_in_world(carved_mesh, world_size, pitch), edited_sources=edited_comparison,
        )
        log(f"[PIPELINE] saved carved mesh: {carved_stl_path}")

    (output_dir / "candidate_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    log(f"[PIPELINE] completed in {time.time() - t0:.2f} seconds")

    return {
        "output_dir": str(output_dir),
        "world_size": world_size,
        "voxel_pitch": pitch,
        "hull_stl_path": hull_stl_path,
        "carved_stl_path": carved_stl_path,
        "hull_summaries": hull_summaries,
        "carved_summaries": carved_summaries,
        "fit_diagnostics": fit,
        "candidate_quality": quality,
    }


def main():
    args = parse_args()

    result = run_pipeline(
        view_paths=args.views,
        world_size=args.world_size,
        grid=args.grid,
        image_size_value=args.image_size,
        optimize_material=args.optimize_material,
        directions=parse_direction_string(args.directions) if args.directions else None,
        opt_iters=args.opt_iters, threshold=args.threshold,
        close_iters=args.close_iters, open_iters=args.open_iters, dilate_iters=args.dilate_iters,
        rank_stl=not args.center_ranking, search_alignment=args.search_alignment,
        search_light_angles=args.search_light_angles, bridge_diameter_voxels=args.bridge_diameter_voxels,
    )
    print_view_metrics("STL shadows against original silhouettes",
                       result["carved_summaries"] or result["hull_summaries"])


if __name__ == "__main__":
    main()
