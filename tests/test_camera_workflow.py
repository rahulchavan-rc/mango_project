import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import use_model

SHAPE_PASS = {
    "detected": True,
    "has_frame_margin": True,
    "area_ratio": 0.25,
    "aspect_ratio": 0.8,
}


class _EncodingCamera:
    def imencode(self, _extension, frame):
        return True, np.asarray(frame, dtype=np.uint8)


class _SequentialGradeModel(use_model.tf.keras.Model):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = iter(predictions)

    @property
    def inputs(self):
        return [object()]

    def predict(self, _inputs, batch_size=None, verbose=0, steps=None, callbacks=None):
        return np.asarray([next(self.predictions)], dtype=np.float32)


class _ValidityModel(use_model.tf.keras.Model):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = probabilities

    def predict(self, _inputs, batch_size=None, verbose=0, steps=None, callbacks=None):
        return np.asarray([self.probabilities], dtype=np.float32)


class _CaptureCamera:
    def __init__(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class _CaptureCV2:
    EVENT_LBUTTONDOWN = 1
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, encoded_image, retake_key=None):
        self.encoded_image = encoded_image
        self.camera = _CaptureCamera()
        self.click_parameter = None
        self.click_callback = None
        self.encode_attempts = 0
        self.retake_key = retake_key
        self.drawn_text = []

    def VideoCapture(self, _camera_index):
        return self.camera

    def namedWindow(self, _name):
        pass

    def setMouseCallback(self, _name, callback, parameter):
        self.click_callback = callback
        self.click_parameter = parameter

    def imencode(self, _extension, _frame):
        self.encode_attempts += 1
        if self.encode_attempts == 1:
            return False, None
        return True, np.frombuffer(self.encoded_image, dtype=np.uint8)

    def rectangle(self, *_args):
        pass

    def putText(self, *_args):
        self.drawn_text.append(_args[1])

    def imshow(self, *_args):
        pass

    def waitKey(self, _delay):
        if self.click_parameter["state"] == "waiting_for_side":
            self.click_callback(
                self.EVENT_LBUTTONDOWN,
                0,
                0,
                0,
                self.click_parameter,
            )
        elif self.click_parameter["state"] == "result":
            if self.retake_key is not None:
                key = self.retake_key
                self.retake_key = None
                return ord(key)
            return ord("q")
        return 0xFF

    def destroyAllWindows(self):
        pass


class CameraWorkflowTests(unittest.TestCase):
    def test_capture_announcements_are_exact_and_ordered(self):
        self.assertEqual(
            [
                use_model._camera_transition_announcement(side)
                for side in use_model.CAMERA_SIDES
            ],
            [
                "Right-side image captured. Show the left side.",
                "Left-side image captured. Show the front.",
                "Front image captured. Show the back.",
                "Back image captured. Please wait for prediction.",
            ],
        )

    def test_atomic_capture_save_overwrites_selected_side(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw" / "right.jpg"
            self.assertEqual(
                use_model._save_camera_capture(np.asarray([1, 2]), path, _EncodingCamera()),
                path,
            )
            self.assertEqual(path.read_bytes(), bytes([1, 2]))
            use_model._save_camera_capture(np.asarray([3, 4]), path, _EncodingCamera())
            self.assertEqual(path.read_bytes(), bytes([3, 4]))

    def test_tied_vote_uses_mean_scores_and_saved_class_order(self):
        class_names = ["grade 4", "grade 1", "grade 3", "grade 2"]
        predicted = ("grade 4", "grade 1", "grade 1", "grade 4")
        scores = (
            {"grade 4": 0.9, "grade 1": 0.1, "grade 3": 0.0, "grade 2": 0.0},
            {"grade 4": 0.2, "grade 1": 0.8, "grade 3": 0.0, "grade 2": 0.0},
            {"grade 4": 0.4, "grade 1": 0.6, "grade 3": 0.0, "grade 2": 0.0},
            {"grade 4": 0.7, "grade 1": 0.3, "grade 3": 0.0, "grade 2": 0.0},
        )
        per_side = {
            side: {
                "status": "VALIDITY_UNAVAILABLE",
                "validity_status": "UNAVAILABLE",
                "predicted_grade": predicted[index],
                "class_scores": scores[index],
            }
            for index, side in enumerate(use_model.CAMERA_SIDES)
        }

        combined = use_model.combine_view_predictions(per_side, class_names)

        self.assertEqual(combined["predicted_grade"], "grade 4")
        self.assertEqual(combined["vote_counts"]["grade 4"], 2)
        self.assertEqual(combined["winning_votes"], 2)
        self.assertEqual(combined["status"], "VALIDITY_UNAVAILABLE")

    def test_non_mango_detector_result_blocks_object_grade(self):
        per_side = {
            side: {
                "status": "NON_MANGO" if side == "left" else "GRADED",
                "predicted_grade": "grade 1",
                "class_scores": {"grade 1": 1.0},
            }
            for side in use_model.CAMERA_SIDES
        }

        combined = use_model.combine_view_predictions(per_side, ["grade 1"])

        self.assertEqual(combined["status"], "NON_MANGO")
        self.assertIsNone(combined["predicted_grade"])
        self.assertEqual(combined["announcement"], "Not a mango—grading unavailable.")

    def test_failed_shape_or_ood_view_prevents_final_grade(self):
        per_side = {
            side: {
                "status": "INVALID" if side == "back" else "VALIDITY_UNAVAILABLE",
                "predicted_grade": None if side == "back" else "grade 1",
                "class_scores": {} if side == "back" else {"grade 1": 1.0},
            }
            for side in use_model.CAMERA_SIDES
        }

        combined = use_model.combine_view_predictions(per_side, ["grade 1"])

        self.assertEqual(combined["status"], "UNCERTAIN")
        self.assertIsNone(combined["predicted_grade"])
        self.assertEqual(combined["rejected_sides"], ["back"])

    def test_ood_filter_uses_saved_feature_centroids(self):
        feature_db = {
            "centroids": {
                "valid": [1.0, 0.0],
                "invalid": [0.0, 1.0],
            },
            "valid_similarity_threshold": 0.65,
        }

        self.assertFalse(use_model.ood_scores([1.0, 0.0], feature_db)[0])
        self.assertTrue(use_model.ood_scores([0.0, 1.0], feature_db)[0])

    def test_validity_predictions_use_saved_label_order(self):
        self.assertEqual(
            use_model._validity_class_indices(["Non-Mango", "Mango"]),
            (1, 0),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "object.jpg"
            Image.new("RGB", (12, 12), color=(10, 20, 30)).save(image_path)
            with patch.object(
                use_model,
                "object_shape_size",
                return_value=SHAPE_PASS,
            ):
                result = use_model.predict_single_image(
                    _ValidityModel([0.9, 0.1]),
                    _SequentialGradeModel([[1.0]]),
                    ["grade 1"],
                    image_path,
                    validity_class_names=["Non-Mango", "Mango"],
                )

        self.assertEqual(result["status"], "NON_MANGO")
        self.assertIsNone(result["predicted_grade"])

    def test_sample_images_process_within_one_isolated_session(self):
        sample_directory = use_model.PROJECT_ROOT / "captured_photos"
        samples = sorted(
            path for path in sample_directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        predictions = [
            [0.80, 0.10, 0.05, 0.05],
            [0.75, 0.15, 0.05, 0.05],
            [0.10, 0.75, 0.10, 0.05],
            [0.70, 0.20, 0.05, 0.05],
        ]
        grade_model = _SequentialGradeModel(predictions)
        with tempfile.TemporaryDirectory() as temporary_directory:
            if not samples:
                sample_directory = Path(temporary_directory) / "synthetic-rgb-samples"
                sample_directory.mkdir()
                for index, side in enumerate(use_model.CAMERA_SIDES):
                    image_path = sample_directory / f"{side}.jpg"
                    Image.new(
                        "RGB",
                        (48, 32),
                        color=(index * 40, 120, 220 - index * 40),
                    ).save(image_path)
                samples = sorted(sample_directory.glob("*.jpg"))
            session_dir = Path(temporary_directory) / "captures" / "sample-session"
            raw_dir = session_dir / "raw"
            raw_dir.mkdir(parents=True)
            image_paths = []
            for index, side in enumerate(use_model.CAMERA_SIDES):
                path = raw_dir / f"{side}.jpg"
                shutil.copyfile(samples[index % len(samples)], path)
                image_paths.append(path)

            with (
                patch.object(use_model, "object_shape_size", return_value=SHAPE_PASS),
                patch.object(use_model, "load_feature_database", return_value=None),
            ):
                result = use_model.predict_captured_views(None, grade_model, image_paths)

            self.assertEqual(result["session_id"], "sample-session")
            self.assertEqual(result["status"], "VALIDITY_UNAVAILABLE")
            self.assertEqual(result["predicted_grade"], "grade 1")
            self.assertEqual(result["vote_counts"]["grade 1"], 3)
            self.assertFalse(result["preprocessing"]["training_compatibility_verified"])
            self.assertTrue((session_dir / "results.json").is_file())
            for side in use_model.CAMERA_SIDES:
                processed = np.load(session_dir / "processed" / f"{side}.npy")
                self.assertEqual(processed.shape, (224, 224, 3))
                self.assertEqual(processed.dtype, np.float32)
                self.assertGreaterEqual(float(processed.min()), 0.0)
                self.assertLessEqual(float(processed.max()), 255.0)

    def test_capture_order_save_failure_retry_session_and_voice(self):
        predictions = [[0.8, 0.1, 0.05, 0.05]] * 8
        grade_model = _SequentialGradeModel(predictions)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shutil.copyfile(
                use_model.PROJECT_ROOT / "class_names.json",
                root / "class_names.json",
            )
            sample = root / "sample.jpg"
            Image.new("RGB", (48, 32), color=(30, 120, 210)).save(sample)
            cv2 = _CaptureCV2(sample.read_bytes(), retake_key="2")
            spoken = []

            class Speech:
                def say(self, message):
                    spoken.append(message)

                def stop(self):
                    pass

            with (
                patch.object(use_model, "PROJECT_ROOT", root),
                patch("importlib.import_module", return_value=cv2),
                patch.object(use_model, "_OfflineSpeech", Speech),
                patch.object(use_model, "object_shape_size", return_value=SHAPE_PASS),
            ):
                paths = use_model.capture_photos(
                    validity_model=None,
                    grade_model=grade_model,
                )

            self.assertEqual(cv2.encode_attempts, 6)
            self.assertEqual([path.stem for path in paths], list(use_model.CAMERA_SIDES))
            self.assertEqual(len({path.parent.parent for path in paths}), 1)
            self.assertTrue(all(path.parent.name == "raw" for path in paths))
            self.assertEqual(
                spoken,
                [
                    use_model._camera_transition_announcement(side)
                    for side in (*use_model.CAMERA_SIDES, "left")
                ],
            )
            self.assertIn("Final result: Grade 1", cv2.drawn_text)
            self.assertFalse(any("Vote tally" in text for text in cv2.drawn_text))
            self.assertFalse(any(text.startswith("Right =") for text in cv2.drawn_text))
            session_dir = paths[0].parent.parent
            self.assertTrue((session_dir / "results.json").is_file())

    def test_camera_result_announcement_only_shows_final_grade(self):
        combined = use_model.combine_view_predictions(
            {
                side: {
                    "status": "VALIDITY_UNAVAILABLE",
                    "validity_status": "UNAVAILABLE",
                    "predicted_grade": "grade 4",
                    "class_scores": {"grade 4": 0.99},
                }
                for side in use_model.CAMERA_SIDES
            },
            ["grade 4"],
        )

        self.assertEqual(combined["announcement"], "Final result: Grade 4")

    def test_prediction_rejects_mixed_sessions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for index, side in enumerate(use_model.CAMERA_SIDES):
                session = root / ("session-a" if index < 3 else "session-b")
                raw_dir = session / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                path = raw_dir / f"{side}.jpg"
                path.touch()
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "same session"):
                use_model.predict_captured_views(None, None, paths)

    def test_offline_speech_worker_preserves_queue_order(self):
        spoken = []

        class Process:
            def __init__(self, command, **_kwargs):
                spoken.append(command[1])

            def wait(self):
                return 0

            def poll(self):
                return 0

        messages = ["Right", "Left", "Front", "Back"]
        with (
            patch.object(use_model.platform, "system", return_value="Darwin"),
            patch.object(use_model.shutil, "which", return_value="/usr/bin/say"),
            patch.object(use_model.subprocess, "Popen", side_effect=Process),
        ):
            speech = use_model._OfflineSpeech()
            for message in messages:
                speech.say(message)
            speech.stop(graceful=True)

        self.assertEqual(spoken, messages)


if __name__ == "__main__":
    unittest.main()
