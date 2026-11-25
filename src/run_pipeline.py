import argparse
from itertools import combinations
import numpy as np
import time

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_raw_shadow_hull, compute_shadow_hull, connected_components_3d
from shadow_source import build_sources
from export_mesh import export_voxels_to_stl
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from projections import pixel_indices, projected_mask
from render import render_shadow
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from postprocess_prune import fast_projection_prune
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

    args = parser.parse_args()

    if len(args.views) > 3:
        parser.error("Please provide one to three silhouette images.")

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


def mesh_triangles_in_world(mesh, world_size, pitch):
    # STL stores index-space coordinates as float32, so match the exported geometry.
    return mesh.triangles.astype(np.float32).astype(float) + (pitch - world_size) / 2


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
    strict, fit = assess_strict_hull(original_sources, voxel_centers)
    for i, view in enumerate(fit["views"]):
        log(f"[FIT] view {i + 1}: {view['no_center_ray']} pixels without a voxel-center ray, "
            f"{view['blocked_by_other_views']} ruled out by other silhouettes")
    for first, second in fit["disjoint_view_pairs"]:
        log(f"[FIT] views {first} and {second} share no strict voxel volume. "
            "Check mask placement and light directions if the edited result loses detail.")

    # optimize the input silhouettes if more than one inputted
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
        )

        for i, src in enumerate(optimized_sources):
            save_mask(
                src.image,
                str(debug_dir / "masks" / "opt" / f"view{i}_optimized_mask.png")
            )

        sources = optimized_sources
    else:
        log("[PIPELINE] skipping optimization (single view)")
        optimized_sources = sources

    log("[PIPELINE] computing shadow hull...")
    hull = compute_shadow_hull(sources, voxel_centers, original_sources=original_sources)

    log(f"[PIPELINE] initial hull voxel count: {int(hull.sum())}")
    log(f"[PIPELINE] initial hull occupancy ratio: {float(hull.mean())}")

    connected_hull = hull
    log("[PIPELINE] pruning hull...")
    hull, post_stats = fast_projection_prune(
        hull,
        voxel_centers,
        optimized_sources=optimized_sources,
        original_sources=original_sources,
        max_passes=prune_passes,
        max_remove_fraction_per_pass=0.15,
        min_face_neighbors=2,
        redundancy_threshold=2.0,
        cleanup_each_pass=True,
        verbose=True,
    )

    log(f"[PIPELINE] fast prune bulk removed: {post_stats['bulk_removed']}")
    log(f"[PIPELINE] fast prune cc removed: {post_stats['cc_removed']}")
    log(f"[PIPELINE] fast prune final hull voxel count: {post_stats['final_voxels']}")

    for i, src in enumerate(original_sources):
        strict_actual = render_shadow(strict, voxel_centers, src)
        hull_actual = render_shadow(hull, voxel_centers, src)
        bridged_actual = render_shadow(connected_hull, voxel_centers, src)
        fit["views"][i].update(
            lost_from_strict_hull=int((src.image & strict_actual & ~hull_actual).sum()),
            lost_at_connectivity_cleanup=int((src.image & bridged_actual & ~hull_actual).sum()),
        )
    (output_dir / "fit_diagnostics.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
    if not hull.any():
        raise ValueError("The silhouettes leave no shared sculpture volume within the allowed edits. "
                         "Check their alignment and light directions.")

    pitch = voxel_pitch(world_size, nx, ny, nz)

    hull_stl_path = str(mesh_dir / "shadow_hull.stl")

    log("[PIPELINE] exporting STL...")
    mesh = export_voxels_to_stl(hull, pitch, hull_stl_path)
    log(f"[PIPELINE] saved raw hull mesh: {hull_stl_path}")
    hull_summaries = simulate_and_save(
        hull, voxel_centers, original_sources, out_dir=str(sim_dir), prefix="hull",
        triangles=mesh_triangles_in_world(mesh, world_size, pitch), edited_sources=optimized_sources,
    )

    carved_stl_path = None
    carved_summaries = None

    if optimize_material:
        log("[PIPELINE] carving material...")
        carved, _ = carve_hollow_shell_strict(
            hull,
            voxel_centers,
            original_sources,
            shell_thickness_voxels=2,
            max_passes=4,
            random_seed=0,
            protect_endcaps=True,
            cleanup_components=True,
            min_component_size=150,
            verbose=True,
        )

        if connected_components_3d(carved)[1] != 1:
            log("[PIPELINE] hollowing would break face connectivity; keeping the solid sculpture")
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
        carved_mesh = export_voxels_to_stl(carved, pitch, carved_stl_path)
        carved_summaries = simulate_and_save(
            carved, voxel_centers, original_sources, out_dir=str(sim_dir), prefix="carved",
            triangles=mesh_triangles_in_world(carved_mesh, world_size, pitch), edited_sources=optimized_sources,
        )
        log(f"[PIPELINE] saved carved mesh: {carved_stl_path}")

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
    )
    print_view_metrics("STL shadows against original silhouettes",
                       result["carved_summaries"] or result["hull_summaries"])


if __name__ == "__main__":
    main()
