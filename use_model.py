from __future__ import annotations

import json
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from textwrap import wrap

import joblib
import numpy as np
from shape_features import extract_shape_features
from shape_gate import MODEL_PATH as SHAPE_GATE_MODEL_PATH, image_shape_features

try:
    import tensorflow as tf
    import keras
    from keras.initializers import GlorotUniform as KerasGlorotUniform, GlorotNormal as KerasGlorotNormal
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow or Keras is not installed in the Python environment you are using.\n"
        "Please run this script with the project conda environment:\n"
        "  conda run -n mango python use_model.py /path/to/image.jpg"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent

VALIDITY_MODEL_CANDIDATES = [
    PROJECT_ROOT / "validity_runs" / "best_validity.keras",
    PROJECT_ROOT / "validity_runs" / "final_validity.keras",
    PROJECT_ROOT / "final_validity.keras",
]


def _strip_legacy_config_keys(config: dict, legacy_keys: tuple[str, ...]) -> dict:
    config = dict(config)
    for key in legacy_keys:
        config.pop(key, None)
    return config


def _patch_keras_initializer_from_config():
    def _strip_legacy_axes(original_from_config):
        def patched(cls, config, *args, **kwargs):
            config = _strip_legacy_config_keys(config, ("input_axes", "output_axes"))
            return original_from_config.__func__(cls, config, *args, **kwargs)
        return classmethod(patched)

    KerasGlorotUniform.from_config = _strip_legacy_axes(KerasGlorotUniform.from_config)
    KerasGlorotNormal.from_config = _strip_legacy_axes(KerasGlorotNormal.from_config)


def _patch_keras_layer_from_config(layer_class, legacy_keys: tuple[str, ...]):
    original_from_config = layer_class.from_config

    def patched(cls, config, *args, **kwargs):
        config = _strip_legacy_config_keys(config, legacy_keys)
        return original_from_config.__func__(cls, config, *args, **kwargs)

    layer_class.from_config = classmethod(patched)


def _patch_keras_batchnorm_from_config():
    from keras.layers import BatchNormalization

    _patch_keras_layer_from_config(BatchNormalization, ("renorm", "renorm_clipping", "renorm_momentum"))


def _patch_keras_dense_from_config():
    from keras.layers import Dense

    _patch_keras_layer_from_config(Dense, ("quantization_config",))


_patch_keras_initializer_from_config()
_patch_keras_batchnorm_from_config()
_patch_keras_dense_from_config()

CUSTOM_LOAD_OBJECTS = {
    "GlorotUniform": KerasGlorotUniform,
    "GlorotNormal": KerasGlorotNormal,
}

GRADE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "mobilenetv3_runs" / "best_model.keras",
    PROJECT_ROOT / "mobilenetv3_runs" / "final_model.keras",
    PROJECT_ROOT / "final_model.keras",
]


def load_validity_model() -> tf.keras.Model | None:
    model_path = next((path for path in VALIDITY_MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        return None
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_LOAD_OBJECTS)
    except Exception as exc:
        raise RuntimeError(f"Could not load mango-validity model at {model_path}: {exc}") from exc
    print(f"Loaded Validity Model: {model_path.name}")
    return model


def load_grade_model() -> tf.keras.Model:
    model_path = next((path for path in GRADE_MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "No trained grade model file was found. Expected one of:\n"
            f"  {GRADE_MODEL_CANDIDATES[0]}\n"
            f"  {GRADE_MODEL_CANDIDATES[1]}"
        )
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_LOAD_OBJECTS, compile=False)
    print(f"Loaded Grade Model: {model_path.name}")
    return model


def load_class_names() -> list[str]:
    class_names_candidates = [
        PROJECT_ROOT / "mobilenetv3_runs" / "class_names.json",
        PROJECT_ROOT / "class_names.json",
    ]
    class_names_path = next((path for path in class_names_candidates if path.exists()), None)
    if class_names_path is None:
        raise FileNotFoundError(f"Class names file not found in: {class_names_candidates}")
    with class_names_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_image(image_path: Path, image_size: int = 224) -> tf.Tensor:
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image


def grade_model_inputs(grade_model: tf.keras.Model, image_tensor: tf.Tensor):
    if len(grade_model.inputs) == 1:
        return image_tensor
    shape_features = extract_shape_features(image_tensor[0])
    return {
        "image": image_tensor,
        "shape_features": tf.expand_dims(shape_features, axis=0),
    }


FEATURE_DATABASE_PATH = PROJECT_ROOT / "extracted_features.json"


@lru_cache(maxsize=1)
def load_shape_gate():
    if not SHAPE_GATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Shape-gate model not found: {SHAPE_GATE_MODEL_PATH}. "
            "Train it with `.venv/bin/python shape_gate.py` before running predictions."
        )
    return joblib.load(SHAPE_GATE_MODEL_PATH)


# A prediction is returned only when every stage has enough evidence.  These
# defaults are deliberately conservative and can be adjusted after validating
# the model on a held-out deployment set.
# The validity model is the primary safeguard against non-mango images.
# It should reject images that are unlike known mango examples while still
# allowing slightly uncertain genuine mangoes through.
VALID_MANGO_THRESHOLD = 0.15
# Require reasonable grade confidence before accepting a predicted mango grade.
GRADE_CONFIDENCE_THRESHOLD = 0.90
# Margin between invalid and valid centroid similarity to treat borderline
# examples as out-of-distribution.
OOD_INVALID_MARGIN = 0.08
MIN_OBJECT_AREA_RATIO = 0.015
MAX_OBJECT_AREA_RATIO = 0.98
MIN_BORDER_MARGIN_PIXELS = 3
CAMERA_SIDES = ("right", "left", "front", "back")
CAMERA_CAPTURE_DELAY_SECONDS = 0.0
CAMERA_STABLE_FRAMES_REQUIRED = 1
MIN_ELIGIBLE_CAMERA_VIEWS = 2


def object_shape_size(path: Path) -> dict[str, float | bool]:
    """Gate on learned silhouette geometry before running appearance models."""
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)

    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=2)
    foreground = distance > max(18.0, float(np.percentile(distance, 70)))

    rows, columns = np.where(foreground)
    if len(rows) == 0:
        return {"detected": False, "area_ratio": 0.0, "aspect_ratio": 0.0}

    area_ratio = float(foreground.mean())
    min_row, max_row = int(rows.min()), int(rows.max())
    min_column, max_column = int(columns.min()), int(columns.max())
    height = max_row - min_row + 1
    width = max_column - min_column + 1
    aspect_ratio = width / max(height, 1)
    edge_contacts = {
        "top": min_row < MIN_BORDER_MARGIN_PIXELS,
        "bottom": max_row >= foreground.shape[0] - MIN_BORDER_MARGIN_PIXELS,
        "left": min_column < MIN_BORDER_MARGIN_PIXELS,
        "right": max_column >= foreground.shape[1] - MIN_BORDER_MARGIN_PIXELS,
    }
    touching_edges = sum(edge_contacts.values())
    has_frame_margin = (
        touching_edges <= 1
        and not edge_contacts["top"]
        and not edge_contacts["left"]
        and not edge_contacts["right"]
    )
    detected = False
    rejection_reason = "object_shape_or_size_check"
    if (
        MIN_OBJECT_AREA_RATIO <= area_ratio <= MAX_OBJECT_AREA_RATIO
        and 0.12 <= aspect_ratio <= 4.0
    ):
        shape_gate = load_shape_gate()
        shape_features = image_shape_features(path).reshape(1, -1)
        detected = int(shape_gate.predict(shape_features)[0]) == 0
        if not detected:
            rejection_reason = "mango_shape_classifier"
    return {
        "detected": detected,
        "rejection_reason": rejection_reason,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "has_frame_margin": has_frame_margin,
    }


def load_feature_database(path: Path | None = None) -> dict | None:
    target_path = path or FEATURE_DATABASE_PATH
    if not target_path.exists():
        return None
    try:
        with target_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Loaded Feature Database: {target_path.name} (Valid: {data.get('num_valid_samples')}, Invalid: {data.get('num_invalid_samples')})")
            return data
    except Exception as e:
        print(f"Warning: Could not load feature database from {target_path}: {e}")
        return None


def extract_cnn_features(grade_model: tf.keras.Model, model_inputs) -> list[float]:
    """Extract internal 576-D feature vector from MobileNetV3 cnn_features layer."""
    try:
        feature_extractor = tf.keras.Model(
            inputs=grade_model.input,
            outputs=grade_model.get_layer("cnn_features").output,
        )
        features = feature_extractor.predict(model_inputs, verbose=0)[0]
        return [round(float(v), 6) for v in features]
    except Exception as e:
        print(f"Warning: Could not extract CNN features: {e}")
        return []


def ood_scores(cnn_features: list[float], feature_db: dict | None) -> tuple[bool, float | None, float | None]:
    """Check whether an embedding resembles the known mango feature space."""
    if not feature_db or not cnn_features or "centroids" not in feature_db:
        return False, None, None

    import numpy as np

    try:
        vector = np.asarray(cnn_features, dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-10
        valid_centroid = np.asarray(feature_db["centroids"]["valid"], dtype=np.float32)
        invalid_centroid = np.asarray(feature_db["centroids"]["invalid"], dtype=np.float32)
        valid_similarity = float(np.dot(vector, valid_centroid))
        invalid_similarity = float(np.dot(vector, invalid_centroid))
        threshold = float(feature_db.get("valid_similarity_threshold", 0.65))

        # Use a margin to avoid borderline embeddings that are too close to the invalid centroid.
        if invalid_similarity >= valid_similarity - OOD_INVALID_MARGIN:
            return True, valid_similarity, invalid_similarity

        return valid_similarity < threshold, valid_similarity, invalid_similarity
    except (KeyError, TypeError, ValueError):
        return False, None, None


def predict_single_image(
    validity_model: tf.keras.Model | None,
    grade_model: tf.keras.Model,
    class_names: list[str],
    path: Path,
    feature_db: dict | None = None,
) -> dict:
    shape_size = object_shape_size(path)

    # Reject shape failures before running any learned appearance classifiers.
    if not shape_size["detected"]:
        return {
            "image_path": str(path),
            "status": "INVALID",
            "rejection_reason": str(shape_size["rejection_reason"]),
            "validity_confidence": 0.0,
            "predicted_grade": "Invalid",
            "grade_confidence": 0.0,
            "class_scores": {},
            "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
            "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
            "ood_valid_similarity": None,
            "ood_invalid_similarity": None,
        }

    image_tensor = preprocess_image(path)

    # 1. Check mango appearance only after the object passes shape checks.
    is_valid = True
    prob_valid = 1.0

    if validity_model is not None:
        val_probs = validity_model.predict(image_tensor, verbose=0)[0]
        model_prob_valid = float(val_probs[0])
        prob_valid = round(model_prob_valid, 6)
        is_valid = model_prob_valid >= VALID_MANGO_THRESHOLD

    if not is_valid:
        return {
            "image_path": str(path),
            "status": "NON_MANGO",
            "rejection_reason": "mango_appearance_check",
            "validity_confidence": round(prob_valid, 6),
            "predicted_grade": "N/A (Non-Mango)",
            "grade_confidence": 0.0,
            "class_scores": {},
            "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
            "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
            "ood_valid_similarity": None,
            "ood_invalid_similarity": None,
        }

    model_inputs = grade_model_inputs(grade_model, image_tensor)
    cnn_features = extract_cnn_features(grade_model, model_inputs)
    is_ood, valid_similarity, invalid_similarity = ood_scores(cnn_features, feature_db)

    if is_ood:
        return {
            "image_path": str(path),
            "status": "OOD",
            "rejection_reason": "strong invalid probability ",
            "validity_confidence": round(prob_valid, 6),
            "predicted_grade": "N/A",
            "grade_confidence": 0.0,
            "class_scores": {},
            "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
            "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
            "ood_valid_similarity": round(valid_similarity, 6) if valid_similarity is not None else None,
            "ood_invalid_similarity": round(invalid_similarity, 6) if invalid_similarity is not None else None,
        }

    # 2. Grade only after both shape and mango-appearance checks pass.
    probabilities = grade_model.predict(model_inputs, verbose=0)[0]
    predicted_index = int(tf.argmax(probabilities).numpy())
    confidence = float(probabilities[predicted_index])

    class_scores = {name: round(float(prob), 6) for name, prob in zip(class_names, probabilities)}

    if confidence < GRADE_CONFIDENCE_THRESHOLD:
        return {
            "image_path": str(path),
            "status": "LOW_CONFIDENCE",
            "rejection_reason": "grade_softmax_threshold",
            "validity_confidence": round(prob_valid, 6),
            "predicted_grade": class_names[predicted_index],
            "grade_confidence": round(confidence, 6),
            "class_scores": class_scores,
            "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
            "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
            "ood_valid_similarity": round(valid_similarity, 6) if valid_similarity is not None else None,
            "ood_invalid_similarity": round(invalid_similarity, 6) if invalid_similarity is not None else None,
        }

    return {
        "image_path": str(path),
        "status": "VALID",
        "validity_confidence": round(prob_valid, 6),
        "predicted_grade": class_names[predicted_index],
        "grade_confidence": round(confidence, 6),
        "class_scores": class_scores,
        "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
        "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
        "ood_valid_similarity": round(valid_similarity, 6) if valid_similarity is not None else None,
        "ood_invalid_similarity": round(invalid_similarity, 6) if invalid_similarity is not None else None,
    }


def predict_captured_views(
    validity_model: tf.keras.Model | None,
    grade_model: tf.keras.Model,
    image_paths: list[Path],
    cancel_event: threading.Event | None = None,
) -> dict:
    if validity_model is None:
        raise RuntimeError(
            "Four-view camera prediction requires the trained mango-validity model; "
            "the check will not be skipped."
        )
    if len(image_paths) != len(CAMERA_SIDES):
        raise ValueError(
            f"Four-view prediction requires exactly one image for each side: {CAMERA_SIDES}."
        )
    path_sides = [path.stem.rsplit("_", 1)[-1] for path in image_paths]
    if tuple(path_sides) != CAMERA_SIDES:
        raise ValueError(
            "Camera views must be provided in Right, Left, Front, Back order, "
            f"one image per side; received {path_sides}."
        )

    class_names = load_class_names()
    feature_db = load_feature_database()
    results = []
    for path in image_paths:
        if cancel_event is not None and cancel_event.is_set():
            return {
                "status": "CANCELLED",
                "predicted_grade": None,
                "grade_confidence": None,
                "eligible_views": 0,
                "combined_scores": {},
                "announcement": "Prediction cancelled.",
            }
        results.append(
            predict_single_image(validity_model, grade_model, class_names, path, feature_db=feature_db)
        )
    if cancel_event is not None and cancel_event.is_set():
        return {
            "status": "CANCELLED",
            "predicted_grade": None,
            "grade_confidence": None,
            "eligible_views": 0,
            "combined_scores": {},
            "announcement": "Prediction cancelled.",
        }

    # LOW_CONFIDENCE views retain their probability vectors as in the existing
    # averaging behavior; shape failures, non-mangoes, and OOD views do not.
    eligible_results = [
        result
        for result in results
        if result["status"] in {"VALID", "LOW_CONFIDENCE"} and result["class_scores"]
    ]

    print("\nPHOTO RESULTS")
    for result in results:
        print(f"{Path(result['image_path']).name}: {result['status']}")

    if len(eligible_results) < MIN_ELIGIBLE_CAMERA_VIEWS:
        print(
            "\nCombined grade unavailable: fewer than "
            f"{MIN_ELIGIBLE_CAMERA_VIEWS} views passed the mango and out-of-distribution checks."
        )
        for result in results:
            rejection_reason = result.get("rejection_reason")
            if rejection_reason:
                print(f"{Path(result['image_path']).name}: {rejection_reason.replace('_', ' ')}")
        return {
            "status": "UNCERTAIN",
            "predicted_grade": None,
            "grade_confidence": None,
            "eligible_views": len(eligible_results),
            "combined_scores": {},
            "announcement": (
                "The result is uncertain. Fewer than two views passed the mango and "
                "out-of-distribution checks."
            ),
        }

    combined_scores = {
        name: float(np.mean([result["class_scores"][name] for result in eligible_results]))
        for name in class_names
    }
    predicted_grade = max(combined_scores, key=combined_scores.get)
    confidence = combined_scores[predicted_grade]
    confident = confidence >= GRADE_CONFIDENCE_THRESHOLD
    print("\nCOMBINED PREDICTION")
    print(f"Eligible views combined: {len(eligible_results)}/{len(CAMERA_SIDES)}")
    print(f"Predicted grade: {predicted_grade.title()}")
    print(f"Grade confidence: {confidence * 100:.2f}%")
    if not confident:
        print("Result: Uncertain — combined confidence is below the threshold.")
        announcement = (
            "The result is uncertain. The combined grade confidence is "
            f"{confidence * 100:.1f} percent, below the required "
            f"{GRADE_CONFIDENCE_THRESHOLD * 100:.0f} percent."
        )
    else:
        print("Result: Confident grade prediction")
        announcement = (
            f"The final grade is {predicted_grade.title()}, with "
            f"{confidence * 100:.1f} percent confidence."
        )
    print("Grade scores:")
    for grade, score in combined_scores.items():
        print(f"  {grade.title()}: {score * 100:.2f}%")
    return {
        "status": "CONFIDENT" if confident else "UNCERTAIN",
        "predicted_grade": predicted_grade,
        "grade_confidence": confidence,
        "eligible_views": len(eligible_results),
        "combined_scores": combined_scores,
        "announcement": announcement,
    }


def predict_path(
    validity_model: tf.keras.Model | None,
    grade_model: tf.keras.Model,
    target_path: str,
    feature_db: dict | None = None,
    save_json_report: bool = False,
) -> None:
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if feature_db is None:
        feature_db = load_feature_database()

    class_names = load_class_names()
    supported_extensions = {".jpg", ".jpeg", ".png"}

    if path.is_file():
        image_paths = [path]
    else:
        image_paths = sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions])
        if not image_paths:
            print(f"No images with supported extensions {supported_extensions} found in {path}")
            return

    print("\n" + "=" * 80)
    print(f"Processing {len(image_paths)} image(s) from: {path}")
    print("=" * 80)

    results = []
    rejected_count = 0
    valid_count = 0

    for i, img_p in enumerate(image_paths, 1):
        res = predict_single_image(validity_model, grade_model, class_names, img_p, feature_db=feature_db)
        results.append(res)

        if res["status"] != "VALID":
            rejected_count += 1
        else:
            valid_count += 1

        if len(image_paths) == 1:
            print("\nPREDICTION")
            print(f"Image: {img_p.name}")
            if res["status"] == "VALID":
                print(f"Result: Mango — {res['predicted_grade'].title()}")
                print(f"Grade confidence: {res['grade_confidence'] * 100:.2f}%")
                
                print("Grade scores:")
                for grade, score in res["class_scores"].items():
                    print(f"  {grade.title()}: {score * 100:.2f}%")
            elif res["status"] == "LOW_CONFIDENCE":
                print(f"Predicted grade: {res['predicted_grade'].title()} (low confidence)")
                print(f"Grade confidence: {res['grade_confidence'] * 100:.2f}%")
                print(f"Reason: {res['rejection_reason'].replace('_', ' ')}")
                print("Grade scores:")
                for grade, score in res["class_scores"].items():
                    print(f"  {grade.title()}: {score * 100:.2f}%")
            else:
                print(f"Result: {res['predicted_grade']}")
                print(f"Reason: {res['rejection_reason'].replace('_', ' ')}")
                if res["class_scores"]:
                    print("Grade scores:")
                    for grade, score in res["class_scores"].items():
                        print(f"  {grade.title()}: {score * 100:.2f}%")
            print("=" * 80)

    if len(image_paths) > 1:
        print(f"\nBATCH PREDICTION SUMMARY:")
        print(f"Total Images: {len(image_paths)} | Graded Mangoes: {valid_count} | Rejected: {rejected_count}")
        print("-" * 80)
        print(f"{'IMAGE NAME':<35} | {'STATUS':<8} | {'CONFIDENCE':<10} | {'PREDICTED GRADE':<20}")
        print("-" * 80)
        for r in results:
            fname = Path(r["image_path"]).name
            if len(fname) > 33:
                fname = fname[:30] + "..."
            status = r["status"]
            if status != "VALID" and r["grade_confidence"] == 0.0:
                conf = "N/A"
            else:
                conf = f"{r['grade_confidence']*100:.1f}%"
            grade = r["predicted_grade"]
            print(f"{fname:<35} | {status:<8} | {conf:<10} | {grade:<20}")
        print("=" * 80)

    # Optional machine-readable report for callers that explicitly request it.
    if save_json_report:
        report_path = PROJECT_ROOT / "prediction_results.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved prediction results JSON report to: {report_path}")


def _detect_object_bounds(frame: np.ndarray, cv2) -> tuple[int, int, int, int] | None:
    pixels = frame.astype(np.float32)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=2)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)
    threshold = max(18.0, float(np.percentile(distance, 70)))
    foreground = (distance > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = (int(value) for value in stats[component])
    area_ratio = area / float(frame.shape[0] * frame.shape[1])
    aspect_ratio = width / max(height, 1)
    if not (MIN_OBJECT_AREA_RATIO <= area_ratio <= MAX_OBJECT_AREA_RATIO):
        return None
    if not (0.12 <= aspect_ratio <= 4.0):
        return None
    return x, y, width, height


def _camera_side_prompt(side: str) -> str:
    return f"Please show the {side} side of the mango. Click when ready."


def _camera_transition_announcement(side: str, next_side: str | None) -> str:
    if next_side is None:
        return "Back side captured. Please wait for the model prediction."
    return f"{side.title()} side captured. Please show the {next_side} side."


def _has_camera_frame_margin(
    bounds: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> bool:
    x, y, width, height = bounds
    frame_height, frame_width = frame_shape[:2]
    margin = max(8, int(min(frame_height, frame_width) * 0.015))
    return (
        x >= margin
        and y >= margin
        and x + width <= frame_width - margin
        and y + height <= frame_height - margin
    )


def _validate_and_save_camera_capture(
    frame: np.ndarray,
    side: str,
    session_id: str,
    capture_dir: Path,
    validity_model: tf.keras.Model,
    cv2,
    cancel_event: threading.Event,
) -> tuple[Path | None, str | None]:
    bounds = _detect_object_bounds(frame, cv2)
    if bounds is None:
        return None, "object_shape_or_size_check"
    if not _has_camera_frame_margin(bounds, frame.shape):
        return None, "object_near_frame_edge"

    x, y, width, height = bounds
    padding_x = max(2, int(width * 0.10))
    padding_y = max(2, int(height * 0.10))
    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(frame.shape[1], x + width + padding_x)
    bottom = min(frame.shape[0], y + height + padding_y)
    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0:
        return None, "empty_crop"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        if not cv2.imwrite(str(temporary_path), crop):
            raise RuntimeError("Could not prepare the camera crop for mango validation.")

        shape_check = object_shape_size(temporary_path)
        if not shape_check["detected"] or not shape_check.get("has_frame_margin", False):
            return None, str(shape_check.get("rejection_reason", "mango_shape_classifier"))

        image_tensor = preprocess_image(temporary_path)
        validity_probabilities = validity_model.predict(image_tensor, verbose=0)[0]
        probability_valid = float(validity_probabilities[0])
        if probability_valid < VALID_MANGO_THRESHOLD:
            return None, "mango_appearance_check"
        if cancel_event.is_set():
            return None, "capture_cancelled"

        capture_dir.mkdir(parents=True, exist_ok=True)
        photo_path = capture_dir / f"camera_capture_{session_id}_{side}.jpg"
        encoded, image_buffer = cv2.imencode(".jpg", crop)
        if not encoded:
            raise RuntimeError("OpenCV could not encode the validated camera crop.")
        with photo_path.open("xb") as output_file:
            output_file.write(image_buffer.tobytes())
        return photo_path, None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class _OfflineSpeech:
    def __init__(self) -> None:
        system = platform.system()
        if system == "Darwin" and shutil.which("say"):
            self.command = "say"
        elif system == "Linux":
            self.command = shutil.which("espeak-ng") or shutil.which("espeak")
        else:
            self.command = None

        self.messages: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.stopping = threading.Event()
        self.worker: threading.Thread | None = None
        if self.command is None:
            print(
                "Offline speech is unavailable. Instructions will remain visible in the preview; "
                "install espeak-ng on Linux to enable speech."
            )
        else:
            self.worker = threading.Thread(target=self._speak_queued_messages, daemon=True)
            self.worker.start()

    def say(self, message: str) -> None:
        if self.command is not None and not self.stopping.is_set():
            self.messages.put(message)

    def _speak_queued_messages(self) -> None:
        while not self.stopping.is_set():
            message = self.messages.get()
            if message is None:
                return
            try:
                with self.lock:
                    if self.stopping.is_set():
                        return
                    process = subprocess.Popen(
                        [self.command, message],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self.process = process
                return_code = process.wait()
                if return_code != 0 and not self.stopping.is_set():
                    print(f"Offline speech exited with status {return_code}; instructions remain visible.")
            except OSError as exc:
                print(f"Offline speech could not be started: {exc}. Instructions remain visible.")
                self.command = None
                return
            finally:
                with self.lock:
                    self.process = None

    def stop(self, graceful: bool = False) -> None:
        if graceful and self.worker is not None:
            self.messages.put(None)
            self.worker.join(timeout=30.0)
            graceful = not self.worker.is_alive()
        self.stopping.set()
        while True:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                break
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
        if not graceful:
            self.messages.put(None)
        if self.worker is not None:
            self.worker.join(timeout=1.0)


def _draw_camera_preview(
    preview: np.ndarray,
    cv2,
    state: str,
    side: str,
    prompt: str,
    notice: str,
    stable_frames: int,
    countdown_seconds: float | None,
    bounds: tuple[int, int, int, int] | None,
) -> None:
    if bounds is not None:
        x, y, width, height = bounds
        color = (0, 220, 0) if stable_frames >= CAMERA_STABLE_FRAMES_REQUIRED else (0, 220, 255)
        cv2.rectangle(preview, (x, y), (x + width, y + height), color, 2)

    height, width = preview.shape[:2]
    lines = [f"View: {side.title()} | State: {state.replace('_', ' ').title()}"]
    if notice:
        lines.extend(wrap(notice, width=max(24, width // 14)))
    lines.extend(wrap(prompt, width=max(24, width // 14)))
    if state == "waiting_for_side":
        lines.append(
            f"Hold still: {min(stable_frames, CAMERA_STABLE_FRAMES_REQUIRED)}"
            f"/{CAMERA_STABLE_FRAMES_REQUIRED} frames"
        )
    if countdown_seconds is not None:
        lines.append(f"Capture in {countdown_seconds:.1f} seconds")
    if state == "prediction" and prompt.startswith("Prediction complete"):
        lines.append("Press q to close.")

    line_height = 26
    overlay_height = min(height, 12 + line_height * len(lines))
    cv2.rectangle(preview, (0, 0), (width, overlay_height), (20, 20, 20), -1)
    for line_number, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (12, 22 + line_number * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def capture_photos(
    camera_index: int = 0,
    validity_model: tf.keras.Model | None = None,
    grade_model: tf.keras.Model | None = None,
) -> list[Path]:
    if validity_model is None:
        raise RuntimeError(
            "Camera capture requires the trained mango-validity model. "
            "Place a validity model in validity_runs/best_validity.keras, "
            "validity_runs/final_validity.keras, or final_validity.keras."
        )
    if grade_model is None:
        raise RuntimeError("Camera capture requires the trained mango-grade model.")
    load_shape_gate()

    try:
        from importlib import import_module

        cv2 = import_module("cv2")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Camera capture requires OpenCV. Install it with: pip install opencv-python"
        ) from exc

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check camera permissions or try another index."
        )

    window_name = "Mango photo capture"
    capture_dir = PROJECT_ROOT / "captured_photos"
    session_id = uuid.uuid4().hex
    captured_paths: list[Path] = []
    side_index = 0
    state = "waiting_for_side"
    prompt = _camera_side_prompt(CAMERA_SIDES[side_index])
    notice = ""
    countdown_deadline: float | None = None
    pending_path: Path | None = None
    work_future: Future | None = None
    cancel_event = threading.Event()
    graceful_speech_stop = False
    previous_bounds = None
    stable_frames = 0
    click_request = {"requested": False, "state": state}

    def request_capture(event, _x, _y, _flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and parameter["state"] == "waiting_for_side":
            parameter["requested"] = True

    speech = _OfflineSpeech()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mango-camera-inference")
    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, request_capture, click_request)
        speech.say("Please show the right side of the mango. Click when ready.")
        print("Camera ready. Capture order: Right -> Left -> Front -> Back -> Prediction. Press 'q' to cancel.")

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")
            bounds = _detect_object_bounds(frame, cv2)
            if bounds is not None:
                x, y, width, height = bounds
                if previous_bounds is not None:
                    old_x, old_y, old_width, old_height = previous_bounds
                    center_shift = np.hypot(
                        (x + width / 2) - (old_x + old_width / 2),
                        (y + height / 2) - (old_y + old_height / 2),
                    )
                    size_change = abs(width - old_width) / max(old_width, 1)
                    size_change += abs(height - old_height) / max(old_height, 1)
                    if center_shift <= max(frame.shape[:2]) * 0.03 and size_change <= 0.3:
                        stable_frames += 1
                    else:
                        stable_frames = 1
                else:
                    stable_frames = 1
                previous_bounds = bounds
            else:
                previous_bounds = None
                stable_frames = 0

            now = time.monotonic()
            countdown_seconds = None
            if state == "countdown" and countdown_deadline is not None:
                countdown_seconds = max(0.0, countdown_deadline - now)
                if countdown_seconds <= 0:
                    countdown_seconds = None
                    if (
                        bounds is None
                        or stable_frames < CAMERA_STABLE_FRAMES_REQUIRED
                        or not _has_camera_frame_margin(bounds, frame.shape)
                    ):
                        state = "waiting_for_side"
                        countdown_deadline = None
                        notice = "Please click again to retry."
                        prompt = "Click the image when ready for the next capture."
                    else:
                        side = CAMERA_SIDES[side_index]
                        state = "validation"
                        countdown_deadline = None
                        notice = "Checking the fresh frame with the mango models..."
                        work_future = executor.submit(
                            _validate_and_save_camera_capture,
                            frame.copy(),
                            side,
                            session_id,
                            capture_dir,
                            validity_model,
                            cv2,
                            cancel_event,
                        )

            if state == "validation" and work_future is not None and work_future.done():
                try:
                    pending_path, rejection_reason = work_future.result()
                except Exception as exc:
                    raise RuntimeError(
                        "Camera mango validation failed. Capture stopped rather than bypassing "
                        "the shape-gate or validity model."
                    ) from exc
                work_future = None
                if pending_path is None:
                    if rejection_reason == "capture_cancelled" and cancel_event.is_set():
                        return []
                    state = "waiting_for_side"
                    notice = "Please click again to retry."
                    prompt = "Click the image when ready for the next capture."
                else:
                    state = "saved_capture"
                    notice = f"{CAMERA_SIDES[side_index].title()} side captured."
                    if side_index < len(CAMERA_SIDES) - 1:
                        prompt = _camera_side_prompt(CAMERA_SIDES[side_index + 1])
                    else:
                        prompt = "Please wait for the model prediction."

            if state == "prediction" and work_future is not None and work_future.done():
                try:
                    prediction = work_future.result()
                except Exception as exc:
                    raise RuntimeError("Four-view model prediction failed.") from exc
                work_future = None
                notice = prediction["announcement"]
                prompt = "Prediction complete. Press q to close."
                speech.say(prediction["announcement"])

            if state == "waiting_for_side" and click_request["requested"]:
                click_request["requested"] = False
                state = "countdown"
                notice = "Hold one mango still and fully inside the frame."
                prompt = "Keep the mango steady and click again for the next capture."
                countdown_deadline = now + CAMERA_CAPTURE_DELAY_SECONDS

            click_request["state"] = state
            preview = frame.copy()
            _draw_camera_preview(
                preview,
                cv2,
                state,
                CAMERA_SIDES[min(side_index, len(CAMERA_SIDES) - 1)],
                prompt,
                notice,
                stable_frames,
                countdown_seconds,
                bounds,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if state == "prediction" and work_future is None and prompt.startswith("Prediction complete"):
                    graceful_speech_stop = True
                    break
                print("Camera capture cancelled.")
                cancel_event.set()
                return []

            if state == "saved_capture" and pending_path is not None:
                captured_paths.append(pending_path)
                print(f"Saved {CAMERA_SIDES[side_index]} view to: {pending_path}")
                next_side_index = side_index + 1
                transition = _camera_transition_announcement(
                    CAMERA_SIDES[side_index],
                    CAMERA_SIDES[next_side_index] if next_side_index < len(CAMERA_SIDES) else None,
                )
                speech.say(transition)
                pending_path = None
                side_index = next_side_index
                if side_index == len(CAMERA_SIDES):
                    state = "prediction"
                    notice = "Back side captured."
                    prompt = "Please wait for the model prediction."
                    work_future = executor.submit(
                        predict_captured_views,
                        validity_model,
                        grade_model,
                        captured_paths.copy(),
                        cancel_event,
                    )
                else:
                    state = "waiting_for_side"
                    notice = f"{CAMERA_SIDES[side_index - 1].title()} side captured."
                    prompt = _camera_side_prompt(CAMERA_SIDES[side_index])

            click_request["state"] = state
    finally:
        cancel_event.set()
        camera.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False, cancel_futures=True)
        speech.stop(graceful=graceful_speech_stop)
    return captured_paths


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Predict mango grade from an image or camera capture.")
    parser.add_argument("path", nargs="?", help="Image file or folder to process")
    parser.add_argument("--camera", action="store_true", help="Capture a photo from the webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    if args.camera and args.path:
        parser.error("provide either an image/folder path or --camera, not both")
    if not args.camera and not args.path:
        parser.error("provide an image/folder path or use --camera")

    v_model = load_validity_model()
    g_model = load_grade_model()

    if args.camera:
        capture_photos(args.camera_index, v_model, g_model)
    else:
        predict_path(v_model, g_model, args.path)
