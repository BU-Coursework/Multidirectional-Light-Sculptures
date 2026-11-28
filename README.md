# Multidirectional Light Sculptures

## Eric Nohara-LeClair, Yingtong (Sophie) Shen, Josh Ilano

The multidirectional light sculpture generation pipeline takes as input one to three reference silhouette images and generates a single component 3D printable voxel sculpture who's orthographic shadows match the inputs'. The pipeline creates the initial shadow hull, optimizes the input silhouettes to reduce inconsistencies among the input silhouettes, prunes redundant voxels, then hollows out the interior of the sculpture to reduce the materials used to print it.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Input Silhouettes](#input-silhouettes)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Web UI](#running-the-web-ui-streamlit-app)
- [Pipeline Stages](#pipeline-stages)
- [Important Parameters](#important-parameters)
- [Output Files](#output-files)
- [Candidate selection and physical checks](#candidate-selection-and-physical-checks)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

## Features

- Shadow hull construction from multiple input silhouettes
- Support for custom multi-direction lighting / custom light directions
- Silhouette optimization to reduce missing shadow regions
- Voxel pruning for material optimization
- Hollow shell generation for material reduction
- Shadow simulation before fabrication
- Mesh export to STL format
- Debug visualizations including voxel slices and shadow comparisons
- Simple Streamlit UI and CLI

## Repository Structure

```bash
C:.
│   .gitignore
│   packages.txt
│   README.md
│   requirements.txt
│
├───assets
│       favicon.png
│
├───inputs
│       view0.png
│       view1.png
│       view2.png
│       view3.png
│       view4.png
│       view5.png
│       view6.png
│       view7.png
│       view8.png
│       view9.png
│
├───outputs
│   │
│   ├───run_2026-05-04_17-17-40
│   │   ├───debug
│   │   │   ├───masks
│   │   │   │   ├───base
│   │   │   │   └───opt
│   │   │   └───slices
│   │   ├───meshes
│   │   │       shadow_hull.stl
│   │   └───sim
│   └───sim
│
├───renders
│
└───src
    │   app.py
    │   carve.py
    │   debug_slices.py
    │   deform.py
    │   distances.py
    │   export_mesh.py
    │   image_io.py
    │   optimize.py
    │   optimize_consistency.py
    │   phylopic_api.py
    │   postprocess_prune.py
    │   projections.py
    │   render.py
    │   render_preview.py
    │   run_pipeline.py
    │   shadow_hull.py
    │   shadow_source.py
    │   simulate.py
    │   voxel_ops.py
    │   warp.py
    │
    └───__pycache__
```

## Requirements

We have listed our dependencies in our requirements.txt file. Please install them before attempting to run locally.

```bash
pip install -r requirements.txt
```

## Input Silhouettes

Input silhouettes are be binary images where:

- Dark pixels represent required shadow coverage
- Light pixels represent empty shadow

You may attempt to use with non binary images; however, the conversion from non binary images to binary images may lose a lot of the detail in the image. For best results, choose input images with clear dark and light regions. For best consistency, choose input images that are similar sizes/shapes.

Put silhouette images inside /inputs and input their paths on the command line when running the pipeline. If using the Streamlit UI, you may upload your own silhouette images or choose silhouettes from the integrated silhouette API.

## Running the Pipeline

Run this command from the root of the project:

```bash
py src/run_pipeline.py inputs/view0.png inputs/view1.png --grid 256 --image-size 256 --optimize-material
```

Example with custom lighting directions:

```bash
python src/run_pipeline.py inputs/view4.png inputs/view6.png --grid 96 --image-size 128 --optimize-material --directions "1,0,0;1,0,1"
```

The --grid, --image-size, --optimize-material, and --directions flags are optional.
Provide one to three silhouette images. By default, the CLI matches the mask resolution to the voxel grid.
Custom directions should be given as semicolon-separated 3D vectors, one per input image.

## Running the Web UI (Streamlit App)

This project also includes an interactive Streamlit interface for generating sculptures without using the command line. To launch the app, run this command from the project root.

### Launch the app

```bash
streamlit run src/app.py
```

### Hosted app

Alternatively, we have hosted the streamlit app. To access, click on this link:

[Click to view hosted app](https://multidirectional-light-sculptures.streamlit.app/)

## Pipeline Stages

1. Load binary silhouette images as binary matrices
2. Configure shadow sources and light directions
3. Check which original target pixels have a voxel-center ray and which rays the other silhouettes rule out
4. Evaluate the strict hull and limited silhouette edits against the original masks, including thin features and holes
5. Route needed connections and prune redundant voxels while preserving original shadow coverage and face connectivity
6. Optionally hollow out the sculpture
7. Export the STL and rasterize its projected triangles against the original masks
8. Save shadow comparisons, fit diagnostics, and debug artifacts to the output run directory

## Important Parameters

Inside `run_pipeline.py`:

- `nx, ny, nz` control voxel grid resolution
- `world_size` controls physical bounding box size
- `iterations` controls silhouette optimization strength
- `sample_per_view` controls optimization runtime vs quality
- `alpha` controls deformation step size
- `sigma` controls displacement smoothing strength
- `max_passes` controls carving aggressiveness
- `shell_thickness_voxels` controls material optimization thickness
- `threshold` controls image binarization
- `directions` specifies custom lighting directions for each view

Higher voxel resolution improves shadow accuracy but increases runtime and memory usage. For best results with minal runtime, we reccomend using a grid and image resolution of 150.

## Output Files

The pipeline generates:

- `outputs/meshes/shadow_hull.stl`
- STL shadow simulation images per view, with missing and extra pixels compared to the original masks
- `fit_diagnostics.json` with per-view counts for sampling, silhouette conflicts, and connectivity losses
- `candidate_quality.json` with the selected light setup, per-view IoU, thin-feature recall, hole preservation, STL volume, and connected core width. Non-dominated alternatives can be downloaded in the app.
- Debug silhouette masks and voxel slices
- Metrics printed to the terminal

## Candidate selection and physical checks

By default, the pipeline projects several STL candidates using the original masks. It weights mean overlap, the weakest view, and preservation of thin features and holes. Candidate geometry can vary by mask edit, pruning level, bridge width, or area sampling when the mask resolution exceeds the voxel grid. The reported connected core width is an erosion-based estimate capped at seven voxels. Printer tolerances and the actual minimum wall or bridge thickness need separate verification.

Candidate ranking also samples four points per target pixel. Background shadows near fine features and inside enclosed openings carry extra cost. The app and `sim_area/` show finer error maps while the original-resolution previews remain comparable across runs. For fragmented hulls the solver tries bridges with alternate endpoints, then tries wider routes that consider the full connector footprint. The candidate table shows connected core width and a Y-up unsupported-voxel estimate for comparing layouts before slicing.

Add `--search-alignment` to try small frame offsets and scale changes. Add `--search-light-angles` when the lights can move. Both keep view 1 fixed and leave the input pixels unchanged. The reported direction, up vector, frame center, and frame size describe the setup for each selected shadow. The CLI flag `--bridge-diameter-voxels 5` tries a wider bridge alternative, and `--center-ranking` reproduces the prior voxel-ranked selection.

Print cleanup runs by default after STL candidate selection. It trims thin features and restores connections or shadow coverage as needed. It keeps a trim only when the resulting STL stays connected and watertight, preserves every previously correct target pixel, adds no new off-target shadow, and preserves the core-width estimate. The untrimmed candidate remains available for comparison. Use `--no-print-cleanup` or clear **Trim thin features for printing** to bypass it. Check the selected STL in your slicer at the intended physical scale to evaluate overhangs and supports.

Run the generated holdouts from the repository root.

```bash
python scripts/benchmark_shadow_quality.py --output outputs/holdout
```

For a physical check, start with the three-view holdout and export the selected STL. Use the selected setup in `candidate_quality.json` to orient the object and lights. Mark the four corners of each predicted frame on a flat, light screen. Lock the camera exposure and position, then photograph each view with only its corresponding light on and the four marks visible. The marks should sit just outside the shadow area. Compare the dark photographed shadow to the pipeline’s white-on-black `*_actual.png`.

```bash
python scripts/compare_physical_shadow.py path/to/hull_view0_actual.png path/to/view0.jpg --corner 110,80 --corner 760,93 --corner 735,710 --corner 98,690 --threshold 128 --output outputs/physical-view0
```

Use your photo’s corner coordinates in clockwise order from the top left. The tool writes a registered photo, binary observed shadow, error overlay, and JSON metrics. Supply `--target path/to/hull_view0_target.png` to also score the photographed shadow against the intended silhouette.

## Known Limitations

- Some silhouette combinations are geometrically incompatible and will lead to many missed pixels in the silhouette. No matter how they are optimized, some geometries just **dont work** together. The pipeline will attempt to generate a structure, but the quality of the output may be compromised
- Mesh surfaces are voxelized and may require smoothing if you desire a smooth sculpture (sanding or computationally)
- Orthographic lighting model only (assumes parallel light rays)
- Runtime increases heavily with grid + image resolution

## Future Improvements

- Mesh smoothing
- Automatic light placement optimization
- Some feedback to the user for their silhouettes **before** attempting to generate the structure (allow for better UX)
