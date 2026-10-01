from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH_CANDIDATES = [
    PROJECT_ROOT / "validity_runs" / "shape_gate.joblib",
    PROJECT_ROOT / "shape_gate.joblib",
]
MODEL_PATH = next((path for path in MODEL_PATH_CANDIDATES if path.exists()), MODEL_PATH_CANDIDATES[0])
IMAGE_SIZE = 224
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROFILE_BINS = 8
FEATURE_COUNT = 6 + PROFILE_BINS * 2


def image_shape_features(image_path: Path) -> np.ndarray:
    """Extract geometry-only descriptors from an object mask."""
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)

    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0)
    background = np.median(border, axis=0)
    contrast = np.linalg.norm(pixels - background, axis=2)
    contrast = ndimage.gaussian_filter(contrast, sigma=2.0)
    foreground = contrast > max(18.0, float(np.percentile(contrast, 70)))
    foreground = ndimage.binary_closing(foreground, structure=np.ones((5, 5), dtype=bool))

    labels, count = ndimage.label(foreground)
    if count == 0:
        return np.zeros(FEATURE_COUNT, dtype=np.float32)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    total_area = int(component_sizes.sum())
    mask = labels == largest_label
    rows, columns = np.where(mask)
    if len(rows) == 0:
        return np.zeros(FEATURE_COUNT, dtype=np.float32)

    min_row, max_row = int(rows.min()), int(rows.max())
    min_column, max_column = int(columns.min()), int(columns.max())
    height = max_row - min_row + 1
    width = max_column - min_column + 1
    area = float(len(rows))
    box_area = float(height * width)

    boundary = mask & ~ndimage.binary_erosion(mask)
    boundary_points = np.argwhere(boundary)
    if len(boundary_points) > 3000:
        boundary_points = boundary_points[:: int(np.ceil(len(boundary_points) / 3000))]
    try:
        hull_area = float(ConvexHull(boundary_points).volume)
        solidity = area / max(hull_area, 1.0)
    except QhullError:
        solidity = 0.0

    centered = np.stack([columns - columns.mean(), rows - rows.mean()], axis=1)
    eigenvalues = np.linalg.eigvalsh(np.cov(centered.T))
    eccentricity = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-8)))

    row_profile = mask[min_row : max_row + 1, min_column : max_column + 1].sum(axis=1)
    column_profile = mask[min_row : max_row + 1, min_column : max_column + 1].sum(axis=0)
    row_indices = np.linspace(0, max(len(row_profile) - 1, 0), PROFILE_BINS).round().astype(int)
    column_indices = np.linspace(0, max(len(column_profile) - 1, 0), PROFILE_BINS).round().astype(int)
    row_profile = row_profile[row_indices] / max(width, 1)
    column_profile = column_profile[column_indices] / max(height, 1)

    base_features = np.asarray(
        [
            area / (IMAGE_SIZE * IMAGE_SIZE),
            width / max(height, 1),
            area / max(box_area, 1.0),
            solidity,
            eccentricity,
            area / max(total_area, 1),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base_features, row_profile, column_profile]).astype(np.float32)


def collect_examples() -> tuple[list[Path], np.ndarray]:
    mango_root = PROJECT_ROOT / "mango"
    valid_paths = sorted(
        path
        for grade in range(1, 5)
        for path in (mango_root / f"grade {grade}").rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "external" in path.parts
    )
    invalid_paths = sorted(
        path
        for path in (mango_root / "invalid").rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not valid_paths or not invalid_paths:
        raise ValueError("Shape-gate training requires images in mango/grade 1-4 and mango/invalid.")
    paths = valid_paths + invalid_paths
    labels = np.concatenate(
        [
            np.zeros(len(valid_paths), dtype=np.int32),
            np.ones(len(invalid_paths), dtype=np.int32),
        ]
    )
    print(f"Shape examples: {len(valid_paths)} mango, {len(invalid_paths)} invalid")
    return paths, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the appearance-independent mango shape gate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths, labels = collect_examples()
    indices = np.arange(len(paths))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    print("Extracting silhouette features...")
    features = np.stack([image_shape_features(paths[index]) for index in indices])
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(features[train_indices], labels[train_indices])
    predictions = model.predict(features[test_indices])
    print(classification_report(labels[test_indices], predictions, target_names=["mango shape", "invalid shape"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved shape gate: {MODEL_PATH}")


if __name__ == "__main__":
    main()
