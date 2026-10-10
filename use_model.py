from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
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
VALIDITY_CLASS_NAMES_CANDIDATES = [
    PROJECT_ROOT / "validity_runs" / "class_names.json",
    PROJECT_ROOT / "validity_class_names.json",
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

GRADE_MODEL_PROJECT_ROOTS = [
    PROJECT_ROOT,
    PROJECT_ROOT.parent / "pro copy 2",
]
GRADE_MODEL_CANDIDATES = [
    model_root / "mobilenetv3_runs" / model_name
    for model_root in GRADE_MODEL_PROJECT_ROOTS
    for model_name in ("best_model.keras", "final_model.keras")
] + [PROJECT_ROOT / "final_model.keras"]
LOADED_GRADE_MODEL_PATH: Path | None = None
LOADED_GRADE_CLASS_NAMES_PATH: Path | None = None
MIN_OBJECT_AREA_RATIO = 0.015
MAX_OBJECT_AREA_RATIO = 0.98
MIN_BORDER_MARGIN_PIXELS = 3
OOD_INVALID_MARGIN = 0.08


def load_validity_model() -> tf.keras.Model | None:
    model_path = next((path for path in VALIDITY_MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        return None
    try:
        class_names = load_validity_class_names()
    except FileNotFoundError as exc:
        print(f"Warning: {exc} Mango validation will be unavailable.")
        return None
    if _validity_class_indices(class_names) is None:
        print(
            "Warning: Validity-model labels must identify both mango and non-mango classes. "
            "Mango validation will be unavailable."
        )
        return None
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_LOAD_OBJECTS)
    except Exception as exc:
        raise RuntimeError(f"Could not load mango-validity model at {model_path}: {exc}") from exc
    print(f"Loaded Validity Model: {model_path.name}")
    return model


def load_validity_class_names() -> list[str]:
    class_names_path = next(
        (path for path in VALIDITY_CLASS_NAMES_CANDIDATES if path.exists()),
        None,
    )
    if class_names_path is None:
        raise FileNotFoundError(
            "Mango-validity class mapping not found. Expected one of: "
            f"{VALIDITY_CLASS_NAMES_CANDIDATES}"
        )
    with class_names_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError(f"Invalid validity class mapping in {class_names_path}.")
    return class_names


def _validity_class_indices(class_names: list[str]) -> tuple[int, int] | None:
    normalized = [name.strip().lower().replace("_", " ").replace("-", " ") for name in class_names]
    mango_labels = {"mango", "valid", "valid mango", "mango valid"}
    non_mango_labels = {"non mango", "not mango", "invalid", "nonmango"}
    mango_indices = [index for index, name in enumerate(normalized) if name in mango_labels]
    non_mango_indices = [index for index, name in enumerate(normalized) if name in non_mango_labels]
    if len(class_names) != 2 or len(mango_indices) != 1 or len(non_mango_indices) != 1:
        return None
    return mango_indices[0], non_mango_indices[0]


def load_grade_model() -> tf.keras.Model:
    global LOADED_GRADE_MODEL_PATH, LOADED_GRADE_CLASS_NAMES_PATH
    model_path = next((path for path in GRADE_MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "No trained grade model file was found. Expected one of:\n"
            + "\n".join(f"  {path}" for path in GRADE_MODEL_CANDIDATES)
        )
    class_names_path = next(
        (
            path
            for path in (
                model_path.parent / "class_names.json",
                model_path.parent.parent / "class_names.json",
            )
            if path.exists()
        ),
        None,
    )
    if class_names_path is None:
        raise FileNotFoundError(
            f"Class mapping for the selected grade model was not found beside {model_path}."
        )
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_LOAD_OBJECTS, compile=False)
    LOADED_GRADE_MODEL_PATH = model_path
    LOADED_GRADE_CLASS_NAMES_PATH = class_names_path
    print(f"Loaded Grade Model: {model_path}")
    return model


def load_class_names() -> list[str]:
    class_names_candidates = (
        [LOADED_GRADE_CLASS_NAMES_PATH]
        if LOADED_GRADE_CLASS_NAMES_PATH is not None
        else [
            PROJECT_ROOT / "mobilenetv3_runs" / "class_names.json",
            PROJECT_ROOT / "class_names.json",
        ]
    )
    class_names_path = next((path for path in class_names_candidates if path.exists()), None)
    if class_names_path is None:
        raise FileNotFoundError(f"Class names file not found in: {class_names_candidates}")
    with class_names_path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(name, str) and name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError(f"Invalid grade class mapping in {class_names_path}.")
    return class_names


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


@lru_cache(maxsize=1)
def load_shape_gate():
    if not SHAPE_GATE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Shape-gate model not found: {SHAPE_GATE_MODEL_PATH}")
    return joblib.load(SHAPE_GATE_MODEL_PATH)


def object_shape_size(path: Path) -> dict[str, float | bool | str]:
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)

    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=2)
    foreground = distance > max(18.0, float(np.percentile(distance, 70)))
    rows, columns = np.where(foreground)
    if len(rows) == 0:
        return {
            "detected": False,
            "rejection_reason": "object_shape_or_size_check",
            "area_ratio": 0.0,
            "aspect_ratio": 0.0,
        }

    area_ratio = float(foreground.mean())
    min_row, max_row = int(rows.min()), int(rows.max())
    min_column, max_column = int(columns.min()), int(columns.max())
    height = max_row - min_row + 1
    width = max_column - min_column + 1
    aspect_ratio = width / max(height, 1)
    edge_contacts = (
        min_row < MIN_BORDER_MARGIN_PIXELS,
        max_row >= foreground.shape[0] - MIN_BORDER_MARGIN_PIXELS,
        min_column < MIN_BORDER_MARGIN_PIXELS,
        max_column >= foreground.shape[1] - MIN_BORDER_MARGIN_PIXELS,
    )
    has_frame_margin = sum(edge_contacts) <= 1 and not any(
        edge_contacts[index] for index in (0, 2, 3)
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
    target_path = path or (PROJECT_ROOT / "extracted_features.json")
    if not target_path.exists():
        print(f"Warning: Feature database not found at {target_path}; OOD checking is unavailable.")
        return None
    with target_path.open("r", encoding="utf-8") as feature_file:
        data = json.load(feature_file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid feature database in {target_path}.")
    return data


def extract_cnn_features(grade_model: tf.keras.Model, model_inputs) -> list[float]:
    feature_extractor = tf.keras.Model(
        inputs=grade_model.input,
        outputs=grade_model.get_layer("cnn_features").output,
    )
    features = feature_extractor.predict(model_inputs, verbose=0)[0]
    return [float(value) for value in features]


def ood_scores(
    cnn_features: list[float],
    feature_db: dict | None,
) -> tuple[bool, float | None, float | None]:
    if not feature_db:
        return False, None, None
    centroids = feature_db.get("centroids")
    if not isinstance(centroids, dict):
        raise ValueError("The feature database is missing its centroids mapping.")

    vector = np.asarray(cnn_features, dtype=np.float32)
    valid_centroid = np.asarray(centroids["valid"], dtype=np.float32)
    invalid_centroid = np.asarray(centroids["invalid"], dtype=np.float32)
    vector /= np.linalg.norm(vector) + 1e-10
    valid_similarity = float(np.dot(vector, valid_centroid))
    invalid_similarity = float(np.dot(vector, invalid_centroid))
    threshold = float(feature_db.get("valid_similarity_threshold", 0.65))
    is_ood = (
        invalid_similarity >= valid_similarity - OOD_INVALID_MARGIN
        or valid_similarity < threshold
    )
    return is_ood, valid_similarity, invalid_similarity


CAMERA_SIDES = ("right", "left", "front", "back")


def predict_single_image(
    validity_model: tf.keras.Model | None,
    grade_model: tf.keras.Model,
    class_names: list[str],
    path: Path,
    validity_class_names: list[str] | None = None,
    image_tensor: tf.Tensor | None = None,
    feature_db: dict | None = None,
) -> dict:
    shape_size = object_shape_size(path)
    if not shape_size["detected"] or not shape_size.get("has_frame_margin", False):
        return {
            "image_path": str(path),
            "status": "INVALID",
            "rejection_reason": str(shape_size["rejection_reason"]),
            "predicted_grade": None,
            "grade_confidence": None,
            "class_scores": {},
            "validity_status": "UNAVAILABLE",
            "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
            "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
        }

    image_tensor = image_tensor if image_tensor is not None else preprocess_image(path)
    validity_status = "UNAVAILABLE"
    if validity_model is not None:
        if validity_class_names is None:
            validity_class_names = load_validity_class_names()
        class_indices = _validity_class_indices(validity_class_names)
        if class_indices is None:
            raise ValueError("The mango-validity class mapping is ambiguous or unsupported.")
        valid_index, non_mango_index = class_indices
        val_probs = validity_model.predict(image_tensor, verbose=0)[0]
        if len(val_probs) != len(validity_class_names):
            raise ValueError("Validity model output does not match its saved class mapping.")
        model_prob_valid = float(val_probs[valid_index])
        validity_status = "AVAILABLE"
        if int(np.argmax(val_probs)) == non_mango_index:
            return {
                "image_path": str(path),
                "status": "NON_MANGO",
                "rejection_reason": "validity_model_predicted_non_mango",
                "validity_confidence": round(model_prob_valid, 6),
                "predicted_grade": None,
                "grade_confidence": None,
                "class_scores": {},
                "validity_status": validity_status,
                "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
                "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
            }

    model_inputs = grade_model_inputs(grade_model, image_tensor)
    valid_similarity = None
    invalid_similarity = None
    if feature_db is not None:
        cnn_features = extract_cnn_features(grade_model, model_inputs)
        is_ood, valid_similarity, invalid_similarity = ood_scores(cnn_features, feature_db)
        if is_ood:
            return {
                "image_path": str(path),
                "status": "OOD",
                "rejection_reason": "out_of_distribution",
                "validity_confidence": (
                    round(model_prob_valid, 6) if validity_model is not None else None
                ),
                "validity_status": validity_status,
                "predicted_grade": None,
                "grade_confidence": None,
                "class_scores": {},
                "ood_valid_similarity": valid_similarity,
                "ood_invalid_similarity": invalid_similarity,
                "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
                "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
            }
    probabilities = grade_model.predict(model_inputs, verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError("Grade model output does not match the saved grade class mapping.")
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    class_scores = {name: float(prob) for name, prob in zip(class_names, probabilities)}
    status = "VALIDITY_UNAVAILABLE" if validity_model is None else "GRADED"
    return {
        "image_path": str(path),
        "status": status,
        "validity_confidence": (
            round(model_prob_valid, 6) if validity_model is not None else None
        ),
        "validity_status": validity_status,
        "predicted_grade": class_names[predicted_index],
        "grade_confidence": confidence,
        "class_scores": class_scores,
        "object_area_ratio": round(float(shape_size["area_ratio"]), 6),
        "object_aspect_ratio": round(float(shape_size["aspect_ratio"]), 6),
        "ood_valid_similarity": (
            round(valid_similarity, 6) if valid_similarity is not None else None
        ),
        "ood_invalid_similarity": (
            round(invalid_similarity, 6) if invalid_similarity is not None else None
        ),
    }


def combine_view_predictions(per_side: dict[str, dict], class_names: list[str]) -> dict:
    if tuple(per_side) != CAMERA_SIDES:
        raise ValueError(f"Predictions must be supplied in {CAMERA_SIDES} order.")
    non_mango_sides = [
        side for side in CAMERA_SIDES if per_side[side]["status"] == "NON_MANGO"
    ]
    vote_counts = {name: 0 for name in class_names}
    if non_mango_sides:
        return {
            "status": "NON_MANGO",
            "predicted_grade": None,
            "vote_counts": vote_counts,
            "announcement": "Not a mango—grading unavailable.",
            "non_mango_sides": non_mango_sides,
        }
    rejected_sides = [
        side
        for side in CAMERA_SIDES
        if not per_side[side].get("class_scores")
        or per_side[side].get("predicted_grade") not in class_names
    ]
    if rejected_sides:
        return {
            "status": "UNCERTAIN",
            "predicted_grade": None,
            "vote_counts": vote_counts,
            "announcement": "Prediction unavailable—one or more views did not pass the mango checks.",
            "rejected_sides": rejected_sides,
        }
    vote_counts: dict[str, int] = {}
    for name in class_names:
        vote_counts[name] = sum(
            per_side[side]["predicted_grade"] == name for side in CAMERA_SIDES
        )
    highest_votes = max(vote_counts.values())
    tied_grades = [name for name in class_names if vote_counts[name] == highest_votes]
    average_scores = {
        name: float(np.mean([per_side[side]["class_scores"][name] for side in CAMERA_SIDES]))
        for name in tied_grades
    }
    final_grade = max(tied_grades, key=average_scores.__getitem__)
    validity_available = all(
        per_side[side].get("validity_status") == "AVAILABLE" for side in CAMERA_SIDES
    )
    limitations = []
    if not validity_available:
        limitations.append("Mango validation unavailable; grade outputs are for mangoes only.")
    limitations.append("External RGB views do not establish internal mango condition.")
    return {
        "status": "GRADED" if validity_available else "VALIDITY_UNAVAILABLE",
        "predicted_grade": final_grade,
        "vote_counts": vote_counts,
        "winning_votes": vote_counts[final_grade],
        "average_model_scores": average_scores,
        "limitations": limitations,
        "announcement": f"Final result: {final_grade.title()}",
    }


def predict_captured_views(
    validity_model: tf.keras.Model | None,
    grade_model: tf.keras.Model | None,
    image_paths: list[Path],
    cancel_event: threading.Event | None = None,
) -> dict:
    if len(image_paths) != len(CAMERA_SIDES):
        raise ValueError(
            f"Four-view prediction requires exactly one image for each side: {CAMERA_SIDES}."
        )
    path_sides = [path.stem.lower() for path in image_paths]
    if tuple(path_sides) != CAMERA_SIDES:
        raise ValueError(
            "Camera views must be provided in Right, Left, Front, Back order, "
            f"one image per side; received {path_sides}."
        )
    session_dirs = {path.parent.parent.resolve() for path in image_paths}
    if len(session_dirs) != 1 or any(path.parent.name != "raw" or not path.is_file() for path in image_paths):
        raise ValueError("All four images must be saved in the same session's raw directory.")

    if grade_model is None:
        grade_model = load_grade_model()
    if validity_model is None:
        validity_model = load_validity_model()
    class_names = load_class_names()
    validity_class_names = load_validity_class_names() if validity_model is not None else None
    feature_db = load_feature_database()
    session_dir = next(iter(session_dirs))
    processed_dir = session_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    per_side: dict[str, dict] = {}
    for side, path in zip(CAMERA_SIDES, image_paths):
        if cancel_event is not None and cancel_event.is_set():
            return {
                "status": "CANCELLED",
                "predicted_grade": None,
                "per_side": per_side,
                "vote_counts": {},
                "announcement": "Prediction cancelled.",
            }
        image_tensor = preprocess_image(path)
        processed_path = processed_dir / f"{side}.npy"
        with processed_path.open("wb") as processed_file:
            np.save(processed_file, image_tensor[0].numpy())
        per_side[side] = predict_single_image(
            validity_model,
            grade_model,
            class_names,
            path,
            validity_class_names=validity_class_names,
            image_tensor=image_tensor,
            feature_db=feature_db,
        )
    if cancel_event is not None and cancel_event.is_set():
        return {
            "status": "CANCELLED",
            "predicted_grade": None,
            "per_side": per_side,
            "vote_counts": {},
            "announcement": "Prediction cancelled.",
        }
    combined = combine_view_predictions(per_side, class_names)
    result = {
        "session_id": session_dir.name,
        "preprocessing": {
            "image_size": [224, 224],
            "resize": "TensorFlow bilinear",
            "channels": "RGB",
            "dtype": "float32",
            "value_range": [0, 255],
            "training_compatibility_verified": False,
        },
        "score_label": "Model scores; not guaranteed accuracy.",
        "per_side": per_side,
        **combined,
    }
    results_path = session_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2)
    print(combined["announcement"])
    return result


def predict_path(
    validity_model: tf.keras.Model | None,
    grade_model: tf.keras.Model,
    target_path: str,
    save_json_report: bool = False,
) -> None:
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    class_names = load_class_names()
    validity_class_names = (
        load_validity_class_names() if validity_model is not None else None
    )
    feature_db = load_feature_database()
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

    for img_p in image_paths:
        res = predict_single_image(
            validity_model,
            grade_model,
            class_names,
            img_p,
            validity_class_names=validity_class_names,
            feature_db=feature_db,
        )
        results.append(res)

        if res["class_scores"]:
            valid_count += 1
        else:
            rejected_count += 1

        if len(image_paths) == 1:
            print("\nPREDICTION")
            print(f"Image: {img_p.name}")
            if res["status"] == "NON_MANGO":
                print("Result: Not a mango—grading unavailable.")
            elif res["class_scores"]:
                print(f"Predicted grade: {res['predicted_grade']}")
                print(f"Model score: {res['grade_confidence'] * 100:.2f}%")
                if res["status"] == "VALIDITY_UNAVAILABLE":
                    print("Limitation: Mango validation unavailable; grade outputs are for mangoes only.")
                print("Grade scores:")
                for grade, score in res["class_scores"].items():
                    print(f"  {grade.title()}: model score {score * 100:.2f}%")
            else:
                print(f"Result: {res['status']}")
            print("=" * 80)

    if len(image_paths) > 1:
        print(f"\nBATCH PREDICTION SUMMARY:")
        print(
            f"Total Images: {len(image_paths)} | Images with grade output: "
            f"{valid_count} | Not graded: {rejected_count}"
        )
        print("-" * 80)
        print(f"{'IMAGE NAME':<35} | {'STATUS':<8} | {'CONFIDENCE':<10} | {'PREDICTED GRADE':<20}")
        print("-" * 80)
        for r in results:
            fname = Path(r["image_path"]).name
            if len(fname) > 33:
                fname = fname[:30] + "..."
            status = r["status"]
            if r["grade_confidence"] is None:
                conf = "N/A"
            else:
                conf = f"{r['grade_confidence']*100:.1f}% model score"
            grade = r["predicted_grade"]
            print(f"{fname:<35} | {status:<8} | {conf:<10} | {grade:<20}")
        print("=" * 80)

    # Optional machine-readable report for callers that explicitly request it.
    if save_json_report:
        report_path = PROJECT_ROOT / "prediction_results.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved prediction results JSON report to: {report_path}")


def _camera_side_prompt(side: str) -> str:
    return f"Show the {side} side of the object, then click the preview."


def _camera_transition_announcement(side: str) -> str:
    announcements = {
        "right": "Right-side image captured. Show the left side.",
        "left": "Left-side image captured. Show the front.",
        "front": "Front image captured. Show the back.",
        "back": "Back image captured. Please wait for prediction.",
    }
    return announcements[side]


def _save_camera_capture(frame: np.ndarray, photo_path: Path, cv2) -> Path:
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    encoded, image_buffer = cv2.imencode(".jpg", frame)
    if not encoded:
        raise RuntimeError("OpenCV could not encode the camera image.")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=photo_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(image_buffer.tobytes())
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, photo_path)
        return photo_path
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
    prediction: dict | None = None,
) -> None:
    height, width = preview.shape[:2]
    is_result = state == "result" and prediction is not None
    lines = [] if is_result else [f"View: {side.title()} | State: {state.replace('_', ' ').title()}"]
    if notice:
        lines.extend(line for line in notice.splitlines() if line)
    lines.extend(wrap(prompt, width=max(24, width // 14)))

    line_height = 26
    lines = lines[:max(1, (height - 12) // line_height)]
    overlay_height = min(height, 12 + line_height * len(lines))
    cv2.rectangle(preview, (0, 0), (width, overlay_height), (20, 20, 20), -1)
    for line_number, line in enumerate(lines):
        is_final_grade = line.startswith("Final result:")
        cv2.putText(
            preview,
            line,
            (12, 22 + line_number * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76 if is_final_grade else 0.58,
            (0, 255, 255) if is_final_grade else (255, 255, 255),
            2 if is_final_grade else 1,
            cv2.LINE_AA,
        )


def capture_photos(
    camera_index: int = 0,
    validity_model: tf.keras.Model | None = None,
    grade_model: tf.keras.Model | None = None,
) -> list[Path]:
    if grade_model is None:
        grade_model = load_grade_model()
    if validity_model is None:
        validity_model = load_validity_model()

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
    session_id = uuid.uuid4().hex
    session_dir = PROJECT_ROOT / "captures" / session_id
    raw_dir = session_dir / "raw"
    captured_paths: dict[str, Path] = {}
    side_index = 0
    state = "waiting_for_side"
    prompt = _camera_side_prompt(CAMERA_SIDES[side_index])
    notice = ""
    work_future: Future | None = None
    prediction: dict | None = None
    cancel_event = threading.Event()
    retake_mode = False
    click_request = {"requested": False, "state": state}

    def request_capture(event, _x, _y, _flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and parameter["state"] == "waiting_for_side":
            parameter["requested"] = True
            parameter["state"] = "saving"

    speech = _OfflineSpeech()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mango-camera-inference")
    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, request_capture, click_request)
        print(
            f"Camera ready. Session {session_id}; capture order: "
            "Right -> Left -> Front -> Back. Click once per side; press q to close."
        )

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")
            if state == "saving" and work_future is not None and work_future.done():
                side = CAMERA_SIDES[side_index]
                try:
                    photo_path = work_future.result()
                except Exception as exc:
                    work_future = None
                    state = "waiting_for_side"
                    notice = f"Could not save {side.title()} image: {exc}"
                    prompt = "Click once to retry this side."
                else:
                    work_future = None
                    captured_paths[side] = photo_path
                    speech.say(_camera_transition_announcement(side))
                    print(f"Saved {side} view to: {photo_path}")
                    notice = f"{side.title()} image saved."
                    if retake_mode:
                        retake_mode = False
                        state = "prediction"
                        prompt = "Please wait for prediction."
                        work_future = executor.submit(
                            predict_captured_views,
                            validity_model,
                            grade_model,
                            [captured_paths[view_side] for view_side in CAMERA_SIDES],
                            cancel_event,
                        )
                    else:
                        side_index += 1
                        if side_index < len(CAMERA_SIDES):
                            state = "waiting_for_side"
                            prompt = _camera_side_prompt(CAMERA_SIDES[side_index])
                        else:
                            state = "prediction"
                            prompt = "Please wait for prediction."
                            work_future = executor.submit(
                                predict_captured_views,
                                validity_model,
                                grade_model,
                                [captured_paths[view_side] for view_side in CAMERA_SIDES],
                                cancel_event,
                            )

            if state == "prediction" and work_future is not None and work_future.done():
                try:
                    prediction = work_future.result()
                except Exception as exc:
                    prediction = None
                    state = "prediction_error"
                    notice = f"Prediction unavailable: {exc}"
                    prompt = "Install the required trained model files, then press p to retry."
                    print(notice)
                else:
                    state = "result"
                    notice = prediction["announcement"]
                    prompt = "Press 1-4 to retake a side, or q to close."
                work_future = None

            if state == "waiting_for_side" and click_request["requested"]:
                click_request["requested"] = False
                side = CAMERA_SIDES[side_index]
                state = "saving"
                notice = f"Saving {side.title()} image..."
                prompt = "The preview remains active."
                work_future = executor.submit(
                    _save_camera_capture,
                    frame.copy(),
                    raw_dir / f"{side}.jpg",
                    cv2,
                )

            click_request["state"] = state
            preview = frame.copy()
            _draw_camera_preview(
                preview,
                cv2,
                state,
                CAMERA_SIDES[min(side_index, len(CAMERA_SIDES) - 1)],
                prompt,
                notice,
                prediction,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cancel_event.set()
                break
            if state in {"result", "prediction_error"} and ord("1") <= key <= ord("4"):
                side_index = key - ord("1")
                retake_mode = True
                prediction = None
                state = "waiting_for_side"
                notice = f"Retaking {CAMERA_SIDES[side_index].title()} side."
                prompt = _camera_side_prompt(CAMERA_SIDES[side_index])
            elif state == "prediction_error" and key == ord("p"):
                state = "prediction"
                notice = "Retrying prediction..."
                prompt = "Please wait for prediction."
                work_future = executor.submit(
                    predict_captured_views,
                    validity_model,
                    grade_model,
                    [captured_paths[view_side] for view_side in CAMERA_SIDES],
                    cancel_event,
                )

            click_request["state"] = state
    finally:
        cancel_event.set()
        camera.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False, cancel_futures=True)
        speech.stop()
    return [captured_paths[side] for side in CAMERA_SIDES if side in captured_paths]


def _report_prediction_unavailable(error: FileNotFoundError) -> None:
    print("\n" + "=" * 72)
    print("PREDICTION UNAVAILABLE")
    print("=" * 72)
    print("No grade was produced because a required trained model or class mapping is missing.")
    print(f"\nDetails: {error}")
    print(
        "\nRestore the existing model and its matching class mapping, then rerun "
        "this command. This program loads model files for inference only; it will "
        "not train or change model weights."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict mango grade from an image or camera capture.")
    parser.add_argument("path", nargs="?", help="Image file or folder to process")
    parser.add_argument("--camera", action="store_true", help="Capture a photo from the webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    if args.camera and args.path:
        parser.error("provide either an image/folder path or --camera, not both")
    if not args.camera and not args.path:
        parser.error("provide an image/folder path or use --camera")

    if args.camera:
        try:
            capture_photos(args.camera_index)
        except FileNotFoundError as exc:
            _report_prediction_unavailable(exc)
            raise SystemExit(2) from exc
    else:
        try:
            v_model = load_validity_model()
            g_model = load_grade_model()
        except FileNotFoundError as exc:
            _report_prediction_unavailable(exc)
            raise SystemExit(2) from exc
        predict_path(v_model, g_model, args.path)
