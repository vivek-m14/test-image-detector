#!/usr/bin/env python3
"""Detect ISO/WB/shutter calibration test shots from a single image, no burst context.

Two independent rule-based scores (see docs/superpowers/specs/2026-08-17-test-shot-detection-design.md):

  Score A -- exposure clipping. Test shots are near-black or near-white AND low
  local detail; clipping fraction alone is not enough (real dim/night photos clip
  too but keep edge structure), so it's gated by a Sobel edge-energy "info" term.
  A continuous low-mean-luminance term rides alongside the hard clipping fraction
  so frames that are clearly too dark without being fully clipped still get caught.

  Score B -- no-subject / generic-content. Catches the "camera fired at a nearby
  wall/rock/surface while adjusting settings" case: no detected face/person AND
  the frame reads as spatially-repetitive generic texture rather than real scene
  structure. Face-absence alone is deliberately NOT sufficient (ring/flower detail
  shots also have no face) -- texture-repetitiveness is the real gate.

p(is_test_shot) = max(score_A, score_B).

Known gap (see spec): the texture-repetitiveness formula under-scores Set11's
rock-texture close-ups -- the primary motivating example for Score B. Needs
iteration on the texture measure itself, not just threshold tuning.

Usage:
    python3 detect_test_shots.py IMAGE.jpg [IMAGE2.jpg ...]
    python3 detect_test_shots.py --validate /path/to/test_shot_images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# -- Score A: exposure clipping --------------------------------------------------
EDGE_REF = 30.0     # typical Sobel edge energy for a detailed photo
MEAN_REF = 70.0     # mean luminance below which darkness starts counting as significant
DARK_THRESH = 12    # pixel value below which a pixel counts as "clipped black"
BRIGHT_THRESH = 244 # pixel value above which a pixel counts as "clipped white"

# -- Score B: no-subject / generic content ---------------------------------------
BLOCK_STD_REF = 25.0  # typical macro-structure (block-mean) variance for a real scene
EDGE_REF2 = 15.0      # minimum edge energy to call something "textured" rather than blank
GRID = 8              # NxN block grid for macro-uniformity measurement
NO_FACE_WEIGHT = 1.0
HAS_FACE_WEIGHT = 0.4

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _load(path: Path, max_dim: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (original BGR image, downsampled grayscale) for scoring."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return img, gray


def _edge_energy(gray: np.ndarray) -> float:
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(sob_x**2 + sob_y**2).mean())


def score_a(gray: np.ndarray) -> tuple[float, float]:
    """Exposure-clipping score. Returns (score, edge_energy) -- edge_energy is
    reused by score_b so callers should compute this once per image."""
    frac_dark = float((gray < DARK_THRESH).mean())
    frac_bright = float((gray > BRIGHT_THRESH).mean())
    mean_l = float(gray.mean())
    edge_energy = _edge_energy(gray)

    info_score = np.clip(edge_energy / EDGE_REF, 0, 1)
    clip_score = max(frac_dark, frac_bright)
    low_light_score = 1 - np.clip(mean_l / MEAN_REF, 0, 1)
    combined_dark = max(clip_score, low_light_score)
    return combined_dark * (1 - info_score), edge_energy


def score_b(orig: np.ndarray, gray: np.ndarray, edge_energy: float) -> float:
    """No-subject / generic-texture score."""
    h, w = gray.shape
    bh, bw = h // GRID, w // GRID
    block_means = np.array([
        gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean()
        for i in range(GRID) for j in range(GRID)
    ])
    macro_uniformity = 1 - np.clip(block_means.std() / BLOCK_STD_REF, 0, 1)
    texture_richness = np.clip(edge_energy / EDGE_REF2, 0, 1)
    texture_repetitiveness = macro_uniformity * texture_richness

    face_input = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), (600, int(600 * h / w)))
    faces = _FACE_CASCADE.detectMultiScale(face_input, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    no_subject = len(faces) == 0

    return texture_repetitiveness * (NO_FACE_WEIGHT if no_subject else HAS_FACE_WEIGHT)


def score_image(path: Path) -> dict:
    orig, gray = _load(path)
    s_a, edge_energy = score_a(gray)
    s_b = score_b(orig, gray, edge_energy)
    return dict(path=str(path), score_a=s_a, score_b=s_b, p=max(s_a, s_b))


def validate(root: Path, threshold: float = 0.3) -> None:
    """Score every image under root/{Set}/*.jpg and root/{Set}/test/*.jpg and
    report AUC / false-positive / recall numbers against the test/normal labels."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for set_dir in sorted(root.iterdir()):
        if not set_dir.is_dir():
            continue
        test_dir = set_dir / "test"
        for f in sorted(set_dir.glob("*.jpg")):
            rows.append((set_dir.name, f.name, False, f))
        if test_dir.is_dir():
            for f in sorted(test_dir.glob("*.jpg")):
                rows.append((set_dir.name, f.name, True, f))

    results = [dict(set=s, name=n, is_test=t, **score_image(f)) for s, n, t, f in rows]

    labels = [1 if r["is_test"] else 0 for r in results]
    p_vals = [r["p"] for r in results]
    a_vals = [r["score_a"] for r in results]
    b_vals = [r["score_b"] for r in results]

    print(f"{'set':22}{'name':16}{'label':5}{'score_a':9}{'score_b':9}{'p':7}")
    for r in results:
        label = "TEST" if r["is_test"] else "norm"
        print(f"{r['set']:22}{r['name']:16}{label:5}{r['score_a']:9.3f}{r['score_b']:9.3f}{r['p']:7.3f}")

    print(f"\nAUC score_a alone:     {roc_auc_score(labels, a_vals):.4f}")
    print(f"AUC score_b alone:     {roc_auc_score(labels, b_vals):.4f}")
    print(f"AUC max(a,b) combined: {roc_auc_score(labels, p_vals):.4f}")

    n_norm = sum(1 for r in results if not r["is_test"])
    n_test = sum(1 for r in results if r["is_test"])
    fp = [r for r in results if not r["is_test"] and r["p"] >= threshold]
    missed = [r for r in results if r["is_test"] and r["p"] < threshold]
    print(f"\nthreshold={threshold}: FP={len(fp)}/{n_norm}  recall={n_test - len(missed)}/{n_test}")
    if fp:
        print("  false positives:", [(r["set"], r["name"]) for r in fp])
    if missed:
        print("  missed:", [(r["set"], r["name"]) for r in missed])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="Image file(s) to score")
    parser.add_argument("--validate", type=Path, help="Run validation report against a test_shot_images-style folder")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.validate:
        validate(args.validate, args.threshold)
        return

    if not args.images:
        sys.exit("Pass image path(s), or --validate a test_shot_images-style folder")

    for path in args.images:
        r = score_image(path)
        print(f"{path}: p={r['p']:.3f}  (score_a={r['score_a']:.3f}, score_b={r['score_b']:.3f})")


if __name__ == "__main__":
    main()
