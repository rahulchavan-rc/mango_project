# Four-view camera workflow

Run from the project directory (using the included virtual environment on
macOS):

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python use_model.py --camera
```

Run software checks with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Use `--camera-index N` to select another OpenCV camera. The live preview remains
active while capturing. Click it once to save each side, in this order:
**Right -> Left -> Front -> Back**. The next side becomes active only after its
image has been written successfully. Any visible object can be captured.

Each run creates an isolated session:

```text
captures/<session_id>/
  raw/right.jpg
  raw/left.jpg
  raw/front.jpg
  raw/back.jpg
  processed/right.npy
  processed/left.npy
  processed/front.npy
  processed/back.npy
  results.json
```

Captured images are not read from the training dataset. During the result screen,
press `1`, `2`, `3`, or `4` to select Right, Left, Front, or Back for a retake,
then click once to replace that side. A successful retake reruns the four-view
prediction. Press `q` to close; any successfully saved files remain in that
session. There is no countdown, automatic capture, or motor-control code in
this project.

## Models and validity limitations

The camera can capture and save images even when models are missing. Prediction
searches for a trained grade model in the current project first:

- `mobilenetv3_runs/best_model.keras`
- `mobilenetv3_runs/final_model.keras`
- `final_model.keras`

The class mapping is loaded from beside the selected model, so class indices
remain paired with that artifact. The checked-in `final_model.keras` and
`class_names.json` are used when no higher-priority MobileNetV3 run is
available. The camera command loads the selected model before opening the
preview; all four saved sides are then predicted through the same preprocessing
and model path as image-file prediction. Each side votes for its predicted
grade; a vote tie is resolved by the mean model score for the tied grades.
The result screen shows only the final result line.

The project also retains its shape-gate model and feature database. These
existing checks run during prediction, not capture: any object can be saved,
but a view rejected by the shape or out-of-distribution checks prevents a
final grade until that side is retaken. A shape rejection is not, by itself,
proof that an object is not a mango.

The grade class order comes from `mobilenetv3_runs/class_names.json` or
`class_names.json`. The checked-in validity model has no companion binary class
mapping, so it is not loaded as a mango/non-mango detector. To enable validity
detection, provide a validity `.keras` file and a companion mapping at
`validity_runs/class_names.json` or `validity_class_names.json`, with explicit
Mango/Non-Mango class labels. Without that mapping, grade output is explicitly
limited to mangoes and is not validated for arbitrary objects. If a configured
validity model identifies a view as non-mango, the result is “Not a
mango—grading unavailable.”

The current inference helper deterministically decodes RGB, stretches each
image to 224×224 with TensorFlow bilinear resizing, then casts to `float32` in
the 0–255 range. The two-input grade model path also computes its existing
shape-feature input from that resized image. This repository does not include
the grade-model training script, so compatibility with its validation pipeline
and any preprocessing embedded in the saved model cannot be independently
reverified here. The result metadata records this as unverified. Supply the
matching training source before treating preprocessing compatibility as
verified.

Four external RGB views cannot establish a mango's internal condition. No
accuracy improvement is claimed; it needs evaluation on appropriate held-out
data.

## Offline speech and hardware checks

- macOS uses its built-in `say` command.
- Linux/Raspberry Pi uses `espeak-ng` or `espeak` when available. On Debian or
  Raspberry Pi OS, install it with `sudo apt install espeak-ng`.
- Speech runs on a background worker, one confirmation at a time. If no speech
  command is installed, visible prompts and results remain available.

Software checks can exercise ordering, atomic saves, session isolation,
retakes, preprocessing, and vote/tie handling. A real camera is still required
to verify camera permissions, preview, focus, and framing. Audible confirmation
requires the relevant local speech command and audio output.
