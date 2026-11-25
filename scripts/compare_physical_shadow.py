"""Register a photographed shadow to a predicted shadow image and score it."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import evaluate_view, make_comparison_image


def register_photo(photo, size, corners=None):
    if corners is None:
        if photo.size != size:
            raise ValueError("Supply four reference corners when the photo is not registered at the prediction size.")
        return photo.convert("L")
    width, height = size
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Provide four finite (x,y) corners in clockwise order from top left.")
    if (corners < 0).any() or (corners >= photo.size).any():
        raise ValueError("Reference corners must lie within the photo.")
    edges = np.roll(corners, -1, axis=0) - corners
    following = np.roll(edges, -1, axis=0)
    if np.any(edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0] <= 0):
        raise ValueError("Reference corners must form a convex clockwise quadrilateral.")
    grid = ((.5, .5), (width - .5, .5), (width - .5, height - .5), (.5, height - .5))
    matrix, mapped = [], []
    for (x, y), (u, v) in zip(grid, corners + .5):
        matrix.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y),
                       (0, 0, 0, x, y, 1, -v * x, -v * y)))
        mapped.extend((u, v))
    coefficients = np.linalg.solve(matrix, mapped)
    return photo.convert("L").transform(size, Image.Transform.PERSPECTIVE,
                                       coefficients, resample=Image.Resampling.BILINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path, help="Pipeline *_actual.png, with white shadow pixels")
    parser.add_argument("photo", type=Path, help="Photo with dark shadows on a light screen")
    parser.add_argument("--corner", action="append", metavar="X,Y",
                        help="Reference corners clockwise from top left, in source photo pixels")
    parser.add_argument("--threshold", type=int, default=128, choices=range(256), metavar="0..255")
    parser.add_argument("--target", type=Path, help="Optional pipeline *_target.png for a second score")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with Image.open(args.prediction) as image:
        predicted = np.asarray(image.convert("L")) >= 128
        size = image.size
    with Image.open(args.photo) as image:
        corners = [tuple(float(value) for value in entry.split(",")) for entry in args.corner] if args.corner else None
        registered = register_photo(image, size, corners)
    observed = np.asarray(registered) < args.threshold
    compared = {"against_prediction": evaluate_view(predicted, observed), "threshold": args.threshold}
    if args.target:
        with Image.open(args.target) as image:
            if image.size != size:
                parser.error("The target and prediction must have the same dimensions.")
            target = np.asarray(image.convert("L")) >= 128
        compared["against_target"] = evaluate_view(target, observed)
    args.output.mkdir(parents=True, exist_ok=True)
    registered.save(args.output / "registered_photo.png")
    Image.fromarray(observed.astype(np.uint8) * 255).save(args.output / "observed_mask.png")
    Image.fromarray(make_comparison_image(predicted, observed)).save(args.output / "prediction_error.png")
    (args.output / "metrics.json").write_text(json.dumps(compared, indent=2), encoding="utf-8")
    print(json.dumps(compared, indent=2))


if __name__ == "__main__":
    main()
