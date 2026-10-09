# Voice-guided mango camera workflow

Run the camera workflow from the project directory:

```sh
python -m pip install -r requirements.txt
python use_model.py --camera
```

Use `--camera-index N` to choose a different OpenCV camera. The preview advances
through **Right -> Left -> Front -> Back -> Prediction**. Click the preview to
start a visible two-second countdown; keep the mango steady and fully inside
the frame. Press `q` at any time to cancel. The preview stays open after the
prediction; press `q` to close it.

Camera capture requires the trained shape-gate, mango-validity, and grade
models. Missing or failing shape-gate/validity checks stop capture rather than
allowing an unchecked image into grading.

## Offline speech

- macOS uses the built-in `say` command; no additional Python package is
  required.
- Raspberry Pi/Linux uses `espeak-ng` (or `espeak` if already installed).
  Install it on Debian/Raspberry Pi OS with:

  ```sh
  sudo apt install espeak-ng
  ```

If no supported speech command is available, the camera workflow continues
with its instructions and results visible in the preview.

## Capture and prediction limitations

Captured files are named `camera_capture_<session-id>_<side>.jpg` in
`captured_photos/`, so later sessions do not overwrite earlier ones. The
four-view average includes only views whose per-image shape, mango-validity,
and out-of-distribution checks pass. A combined grade is announced as
confident only when at least two views are eligible and the averaged top-grade
probability meets the configured confidence threshold.

The foreground bounds and shape gate are silhouette/geometry checks; they do
not guarantee mango recognition or reliable detection of multiple objects.
The trained validity model checks appearance but is limited by its training
data. Right/left/front/back are user-guided labels, not automatically detected
orientations.

## Validation

Automated code checks can verify workflow state transitions, countdown timing,
retry and duplicate-click handling, cancellation, unavailable speech, and
prediction eligibility/confidence. A real camera, camera permissions, and
local speech command are still needed for hardware verification of preview,
focus/lighting, audible prompts, and actual mango crops.
