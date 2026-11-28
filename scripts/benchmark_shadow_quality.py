"""Compare mesh ranking against the prior solver on holdout masks or a saved audit."""

import argparse
from contextlib import redirect_stdout
import csv
from html import escape
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_mesh import mesh_from_voxels, mesh_triangles_in_world
from light_setup import ViewAdjustment
from render import render_mesh_shadow
from run_pipeline import run_pipeline
from shadow_source import build_sources


def holdout_masks(root):
    root.mkdir(parents=True, exist_ok=True)
    size = 96

    def draw(name, marks):
        path = root / f"{name}.png"
        image = Image.new("L", (size, size), 255)
        marks(ImageDraw.Draw(image))
        image.save(path)
        return path

    def nested(draw):
        draw.ellipse((9, 13, 88, 84), fill=0)
        draw.ellipse((16, 20, 81, 77), fill=255)
        draw.rounded_rectangle((25, 28, 72, 68), radius=8, fill=0)
        draw.rounded_rectangle((33, 33, 64, 63), radius=5, fill=255)
        draw.polygon([(38, 58), (63, 44), (49, 37)], fill=0)
        draw.line([(15, 47), (26, 47)], fill=0, width=2)

    def grille(draw):
        draw.rounded_rectangle((9, 17, 85, 81), radius=7, fill=0)
        for x, width in ((18, 2), (25, 3), (34, 4), (45, 6), (58, 8), (73, 4)):
            draw.rectangle((x, 24, x + width, 72), fill=255)

    def satellites(draw):
        draw.polygon([(20, 46), (51, 32), (77, 48), (53, 64)], fill=0)
        for x, y, r in ((14, 29, 4), (49, 18, 5), (78, 20, 3), (83, 65, 4),
                        (38, 79, 6), (19, 65, 3)):
            draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

    alignment = []
    for top in (25, 36):
        alignment.append(draw(f"registration-{top}", lambda brush: brush.rounded_rectangle(
            (22, top, 74, top + 37), radius=8, fill=0)))
    rings = draw("nested-cutouts", nested)
    slots = draw("narrow-slots", grille)
    points = draw("satellites", satellites)

    volume = np.zeros((64, 64, 64), dtype=bool)
    volume[18:47, 22:45, 28:39] = True
    volume[35:42, 17:54, 16:49] = True
    volume[13:39, 34:40, 32:49] = True
    volume[38:52, 20:27, 20:42] = True
    mesh = mesh_from_voxels(volume, 80 / 64)
    triangles = mesh_triangles_in_world(mesh, 80, 80 / 64)
    sources = build_sources([np.zeros((size, size), dtype=bool)] * 2, 80)
    angled_paths = []
    for i, source in enumerate(sources):
        if i:
            source = ViewAdjustment(pitch=12).apply(source)
        path = root / f"oblique-view-{i + 1}.png"
        Image.fromarray(np.where(render_mesh_shadow(triangles, source), 0, 255).astype(np.uint8)).save(path)
        angled_paths.append(path)

    return [
        ("nested-cutouts", [rings], 64, 64, {}),
        ("slots-and-islands", [slots, points], 64, 64, {}),
        ("three-detail-types", [slots, rings, points], 64, 64, {}),
        ("undersampled-slots", [slots], 48, 96, {}),
        ("shifted-pair", alignment, 64, 64, {"search_alignment": True}),
        ("oblique-pair", angled_paths, 64, 64, {"search_light_angles": True}),
    ]


def execute(output, paths, grid, size, seed=0, **options):
    np.random.seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        result = run_pipeline(paths, grid=grid, image_size_value=size, world_size=80,
                              output_dir=output, **options)
    quality = result["candidate_quality"]
    selected = quality["candidates"][quality["selected"]]
    shadows = result["carved_summaries"] or result["hull_summaries"]
    prefix = "carved" if result["carved_summaries"] else "hull"
    result = {"mean": mean(view["iou"] for view in shadows), "worst": min(view["iou"] for view in shadows),
            "candidate": selected["name"], "sim": output / "sim", "prefix": prefix,
            "voxels": selected["occupied_voxels"], "core": selected["connected_core_width"],
            "quality_file": output / "candidate_quality.json",
            "stl_file": Path(result["carved_stl_path"] or result["hull_stl_path"])}
    adjusted = next((item for item in quality["candidates"]
                     if item["adjusted_setup"] and item.get("sim_dir")), None)
    if adjusted:
        result["setup_alternative"] = {
            "mean": adjusted["mean_iou"], "worst": adjusted["worst_iou"], "candidate": adjusted["name"],
            "sim": Path(adjusted["sim_dir"]), "prefix": "hull", "voxels": adjusted["occupied_voxels"],
            "core": adjusted["connected_core_width"], "quality_file": result["quality_file"],
            "stl_file": Path(adjusted["stl_path"]),
        }
    return result


def write_report(root, rows, details):
    def url(path):
        from os.path import relpath
        return escape(Path(relpath(path.resolve(), root.resolve())).as_posix(), quote=True)

    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fields = ("case", "variant", "before_mean", "after_mean", "before_worst", "after_worst", "candidate")
    stats = "".join(
        '<tr>' + ''.join(f'<td>{escape(format(row[key], ".1%") if isinstance(row[key], float) else row[key])}</td>'
                         for key in fields) + '</tr>' for row in rows
    )
    galleries = []
    for case, version, before, after, view_count in details:
        views = []
        for i in range(view_count):
            images = [("Target", after["sim"] / f'{after["prefix"]}_view{i}_target.png')]
            for label, record in (("Before", before), ("After", after)):
                path = record["sim"] / f'{record["prefix"]}_view{i}_actual.png'
                images.append((label, path))
                images.append((label + " error", path.with_name(path.name.replace("_actual", "_comparison"))))
            views.append(f'<h3>View {i+1}</h3><div class="images">' + "".join(
                f'<figure><img loading="lazy" src="{url(path)}" alt="{label}"><figcaption>{label}</figcaption></figure>'
                for label, path in images
            ) + '</div>')
        galleries.append(f'<details><summary>{escape(case)} · {escape(version)}</summary>'
                         f'<p><a href="{url(after["quality_file"])}">Candidate metrics and setup</a>'
                         f' · <a href="{url(after["stl_file"])}">STL</a></p>'
                         + "".join(views) + "</details>")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Shadow candidate benchmark</title>
<style>body{{max-width:1040px;margin:auto;padding:30px;background:#f5f4ee;color:#172929;font:16px system-ui}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ccd5d0}}
details{{background:white;margin:8px 0;padding:12px}}summary{{cursor:pointer;font-weight:bold}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px}}
figure{{margin:0}}img{{width:100%;image-rendering:pixelated}} a{{color:#225c69}}</style>
<h1>Shadow candidate benchmark</h1><p>IoU compares exported STL shadows with the input silhouettes. Setup trials keep the same masks and permit physical light or frame placement changes. Red is missing coverage and blue is extra shadow.</p>
<p><a href="comparison.csv">Metrics CSV</a></p><div style="overflow:auto"><table><thead><tr>
<th>Case</th><th>Trial</th><th>Mean before</th><th>Mean after</th><th>Worst before</th><th>Worst after</th><th>Selected hull</th>
</tr></thead><tbody>{stats}</tbody></table></div>{''.join(galleries)}</html>'''
    (root / "report.html").write_text(text, encoding="utf-8")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--legacy-audit", type=Path, help="Path to the earlier shadow-quality-improvements audit")
    options = args.parse_args()
    root = options.output.resolve()
    rows, details = [], []

    def record(case, variant, before, after, count):
        rows.append({"case": case, "variant": variant, "before_mean": before["mean"], "after_mean": after["mean"],
                     "before_worst": before["worst"], "after_worst": after["worst"], "candidate": after["candidate"],
                     "after_occupied_voxels": after["voxels"], "after_connected_core_width": after["core"]})
        details.append((case, variant, before, after, count))
        print(f"{case} ({variant}): {before['mean']:.1%} to {after['mean']:.1%}", flush=True)

    if options.legacy_audit:
        legacy = options.legacy_audit.resolve()
        for file in sorted((legacy / "runs").glob("*/result.json")):
            old = json.loads(file.read_text(encoding="utf-8"))
            case = old["id"]
            paths = [legacy / "inputs" / f"{asset}.png" for asset in old["assets"]]
            vertices = old["meshes"]["hollow" if old["hollow"] else "final"]["views"]
            before = {"mean": mean(view["iou"] for view in vertices), "worst": min(view["iou"] for view in vertices),
                      "sim": file.parent / "sim", "prefix": "carved" if old["hollow"] else "hull"}
            after = execute(root / case, paths, old["grid"], old["image_size"], old["seed"],
                            prune_passes=old["prune_passes"], optimize_material=old["hollow"],
                            opt_iters=6 if old.get("use_optimizer", "-no-opt" not in case) else 0)
            record(case, "STL ranking", before, after, len(paths))
    else:
        for case, paths, grid, size, search in holdout_masks(root / "inputs"):
            before = execute(root / case / "center", paths, grid, size, rank_stl=False)
            after = execute(root / case / "stl", paths, grid, size)
            record(case, "STL ranking", before, after, len(paths))
            if search:
                adjusted = execute(root / case / "setup", paths, grid, size, **search)
                record(case, "setup search", after, adjusted, len(paths))
                if "setup_alternative" in adjusted:
                    record(case, "setup alternative", after, adjusted["setup_alternative"], len(paths))
    write_report(root, rows, details)


if __name__ == "__main__":
    main()
