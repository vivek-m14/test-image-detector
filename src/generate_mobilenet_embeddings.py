#!/usr/bin/env python3
"""Regenerate an album's mobilenet.npz to match the backend's own embedding computation.

The backend (modelling/src/cradles/with_selects/mod.rs::get_image_embedding +
modelling/src/children/image_embedding.rs) does exactly this to each image:

    cv2 decode -> BGR2RGB -> resize(224, 224, INTER_LINEAR) -> x/127.5 - 1 -> mobilenet
    -> 0.3*avg_pool + 0.7*max_pool -> L2 normalize

PIL/torchvision's `Resize(BILINEAR)` anti-aliases on downscale; cv2's `INTER_LINEAR` does
not. At the >10x downscale ratios common in these albums (e.g. 2560x1707 -> 224x224) that
difference is large enough to move CTT's own scene-similarity graph, and occasionally its
picks. This script mirrors the backend's actual preprocessing, not the "equivalent"
torchvision transform used by scripts/generate_dino_emb.py's `--model mobilenet` path.

Usage:
    export CTT_ENV=mac
    python scripts/generate_mobilenet_embeddings.py --slug abbyandgarret
    python scripts/generate_mobilenet_embeddings.py --slug abbyandgarret --out /tmp/fixed.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_CTT_V2_FINAL = _SCRIPTS_DIR.parent
_WORKSPACE = _CTT_V2_FINAL.parent  # sibling repos (local/, dataset/) live here -- see CLAUDE.md
_ALBUMS_YAML = _WORKSPACE / "config" / "albums.yaml"
_DEFAULT_MODEL_PATH = _WORKSPACE / "local" / "assets" / "culling" / "starscourge_radahn.onnx"


def album_paths(slug: str) -> tuple[Path, Path, Path]:
    """Return (raw JPEG folder, scores/few.json, embed/mobilenet.npz) for a slug."""
    cfg = yaml.safe_load(_ALBUMS_YAML.read_text())
    env = os.environ.get("CTT_ENV")
    if not env:
        sys.exit("Set CTT_ENV (mac|pod) -- see config/albums.yaml's `environments` block.")
    if slug not in cfg["albums"]:
        sys.exit(f"Unknown slug {slug!r} (not in {_ALBUMS_YAML})")

    dataset_root = Path(cfg["environments"][env]["dataset_root"])
    runs_root = Path(cfg["environments"][env]["runs_root"])
    album_folder = dataset_root / cfg["albums"][slug]["folder"]
    scores_few = runs_root / slug / "scores" / "few.json"
    mobilenet_npz = runs_root / slug / "embed" / "mobilenet.npz"
    return album_folder, scores_few, mobilenet_npz


def image_filenames(scores_few_json: Path) -> list[str]:
    """few.json is a list of clusters, each a list of image records with a `path`."""
    clusters = json.loads(scores_few_json.read_text())
    return [os.path.basename(img["path"]) for cluster in clusters for img in cluster]


def onnx_providers() -> list:
    """CPU only, deliberately. CoreML's compute-unit dispatch (ANE/GPU/CPU) is not
    reproducible across process launches -- the same model and input can land on a
    different internal path from one run to the next, which measured up to ~0.5%
    cosine drift here. Deterministic parity with the backend is the entire point of
    this script, so it trades CoreML's speed for CPU's repeatability."""
    return ["CPUExecutionProvider"]


def preprocess(image_path: Path) -> np.ndarray:
    """cv2 decode -> RGB -> 224x224 INTER_LINEAR resize -> [-1, 1] float32, NHWC."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 127.5 - 1.0)[None, ...]


def blend_and_normalize(avg_pool: np.ndarray, max_pool: np.ndarray) -> np.ndarray:
    """The production blend weights, then L2-normalized (matches the backend's
    `ctt_adapter` write-back, which is what mobilenet.npz has always stored)."""
    embedding = avg_pool.reshape(-1) * 0.3 + max_pool.reshape(-1) * 0.7
    return (embedding / np.linalg.norm(embedding)).astype(np.float32)


def embed_one(session: ort.InferenceSession, image_path: Path) -> np.ndarray:
    outputs = session.run(None, {"input": preprocess(image_path)})
    by_name = dict(zip((o.name for o in session.get_outputs()), outputs))
    return blend_and_normalize(
        by_name["global_average_pooling2d_4"], by_name["global_max_pooling2d_3"]
    )


def generate(album_folder: Path, filenames: list[str], model_path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=onnx_providers())
    embeddings = np.zeros((len(filenames), 1024), dtype=np.float32)
    missing = []

    for i, name in enumerate(filenames):
        try:
            embeddings[i] = embed_one(session, album_folder / name)
        except FileNotFoundError:
            missing.append(name)
        if (i + 1) % 200 == 0 or i + 1 == len(filenames):
            print(f"  {i + 1}/{len(filenames)} images", file=sys.stderr)

    if missing:
        pct = 100.0 * len(missing) / len(filenames)
        if pct > 5.0:
            sys.exit(f"{len(missing)}/{len(filenames)} images ({pct:.1f}%) missing -- aborting")
        print(f"WARNING: {len(missing)} images missing, wrote zero vectors for them", file=sys.stderr)

    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slug", required=True, help="Album slug, must exist in config/albums.yaml")
    parser.add_argument("--out", type=Path, help="Output .npz path (default: the album's own mobilenet.npz)")
    parser.add_argument("--model-path", type=Path, default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    album_folder, scores_few, default_out = album_paths(args.slug)
    out_path = args.out or default_out

    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} already exists -- pass --force to overwrite")
    if not album_folder.is_dir():
        sys.exit(f"Album folder not found: {album_folder}")
    if not scores_few.is_file():
        sys.exit(f"Missing {scores_few} -- run backend culling first")

    filenames = image_filenames(scores_few)
    print(f"{args.slug}: {len(filenames)} images from {scores_few}", file=sys.stderr)

    embeddings = generate(album_folder, filenames, args.model_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), embeddings=embeddings, names=np.asarray(filenames, dtype=str))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
