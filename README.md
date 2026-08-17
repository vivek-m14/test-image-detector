# test-image-detector

Detects ISO/WB/shutter calibration test shots — the throwaway frames a photographer
fires before a real session/pose while dialing in settings — from a single image,
with no access to burst/neighboring frames.

## Usage

```
python3 src/detect_test_shots.py IMAGE.jpg [IMAGE2.jpg ...]
python3 src/detect_test_shots.py --validate /path/to/test_shot_images
```

`--validate` expects a folder of `{Set}/*.jpg` (normal images) and `{Set}/test/*.jpg`
(labeled test shots), and prints per-image scores plus AUC/false-positive/recall
numbers against those labels.

## How it works

`p(is_test_shot) = max(score_A, score_B)`:

- **Score A** — exposure clipping: near-black/near-white AND low local detail
  (a hard clipping fraction alone would also flag real dark/night photos, so it's
  gated by Sobel edge energy).
- **Score B** — no-subject/generic content: no detected face/person AND the frame
  reads as spatially-repetitive generic texture (e.g. the camera fired at a nearby
  wall or surface) rather than real scene structure.

Full design rationale, validation results, and known gaps: see
[`docs/superpowers/specs/2026-08-17-test-shot-detection-design.md`](docs/superpowers/specs/2026-08-17-test-shot-detection-design.md).

Also in `src/`: `generate_mobilenet_embeddings.py` (mirrors the production backend's
mobilenet embedding computation) and `visual_rejection_sample.py` (probes an
existing TinyViT visual-rejection model as a candidate signal — found to be too
confounded by dark/low-light scenes to use standalone; see the spec for numbers).
