#!/usr/bin/env python3
"""Run TinyViT visual rejection on 3 worst-offender 1-star albums."""

import json, os, sys, time, csv
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import onnxruntime as ort

WORKSPACE = Path("/Volumes/My Passport/vk_data/culling/code")
MODEL_PATH = WORKSPACE / "culling_rejection_model_visual/onnx_model/tiny_vit_3class/model.onnx"
CONFIG_PATH = WORKSPACE / "config/albums.yaml"
RUNS_ROOT = WORKSPACE / "local/runs"
OUTPUT_DIR = RUNS_ROOT / "visual_rejection_analysis"
BATCH_SIZE = 32
CLASS_NAMES = ["unrated", "one_star", "four_five"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAMPLE_SLUGS = ["dhrupah-1star_selection", "amwodwinds-1star", "rachelsamuel-1star"]


def preprocess(path: str):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    scale = 224 / min(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    a = np.array(img, dtype=np.float32) / 255.0
    a = ((a - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return a


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = yaml.safe_load(open(CONFIG_PATH))
    mac_root = cfg["environments"]["mac"]["dataset_root"]

    print(f"Loading model …")
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    all_img = []
    all_clust = []
    t0 = time.time()

    for slug in SAMPLE_SLUGS:
        folder = cfg["albums"][slug]["folder"]
        img_dir = os.path.join(mac_root, folder)
        clusters = json.load(open(RUNS_ROOT / slug / "scores" / "few.json"))
        exif_raw = json.load(open(RUNS_ROOT / slug / "exif.json"))
        exif = exif_raw.get("images", exif_raw)

        n_done, n_miss = 0, 0
        for ci, cluster in enumerate(clusters):
            arrs, fnames, stars_list = [], [], []
            for im in cluster:
                fn = os.path.basename(im["path"])
                lp = os.path.join(img_dir, fn)
                st = exif.get(fn, {}).get("stars", None)
                st = st if isinstance(st, (int, float)) else -1
                a = preprocess(lp)
                if a is not None:
                    arrs.append(a)
                    fnames.append(fn)
                    stars_list.append(st)
                else:
                    n_miss += 1
            n_done += len(cluster)

            if not arrs:
                continue

            probs_parts = []
            for b in range(0, len(arrs), BATCH_SIZE):
                batch = np.stack(arrs[b:b+BATCH_SIZE])
                logits = sess.run(None, {inp_name: batch})[0]
                probs_parts.append(softmax(logits))
            probs = np.concatenate(probs_parts)

            c_p1, c_preds, c_stars = [], [], []
            for i in range(len(fnames)):
                p = probs[i]
                pred = CLASS_NAMES[np.argmax(p)]
                all_img.append(dict(slug=slug, filename=fnames[i], stars=stars_list[i],
                                    p_unrated=float(p[0]), p_onestar=float(p[1]),
                                    p_fourfive=float(p[2]), pred_class=pred))
                c_p1.append(float(p[1]))
                c_preds.append(pred)
                c_stars.append(stars_list[i])

            rated = [s for s in c_stars if s >= 0]
            is_junk = all(s <= 1 for s in rated) if rated else False
            has_45 = any(s >= 4 for s in c_stars)
            all_clust.append(dict(
                slug=slug, cluster_idx=ci, size=len(c_p1),
                is_junk=is_junk, has_45=has_45,
                max_p_onestar=max(c_p1), mean_p_onestar=float(np.mean(c_p1)),
                frac_pred_onestar=sum(1 for x in c_preds if x == "one_star") / len(c_preds),
                max_p_fourfive=float(max(probs[:, 2])),
            ))

            if ci % 200 == 0:
                print(f"  {slug} cluster {ci}/{len(clusters)}  ({time.time()-t0:.0f}s)")

        print(f"  {slug}: done — {n_done} images, {n_miss} missing  ({time.time()-t0:.0f}s)")

    # ── save CSVs ──
    with open(OUTPUT_DIR / "sample_cluster_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_clust[0].keys()))
        w.writeheader(); w.writerows(all_clust)

    with open(OUTPUT_DIR / "sample_image_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_img[0].keys()))
        w.writeheader(); w.writerows(all_img)

    # ── analysis ──
    out = []
    def p(s=""): out.append(s); print(s)

    p("=" * 70)
    p("VISUAL REJECTION — SAMPLE (3 worst-offender albums)")
    p("=" * 70)
    p(f"Images scored: {len(all_img)}  |  Clusters: {len(all_clust)}  |  Time: {time.time()-t0:.0f}s")

    # A) image-level
    p("\n" + "─" * 70)
    p("A) IMAGE-LEVEL")
    p("─" * 70)
    by_star = {}
    for r in all_img:
        by_star.setdefault(r["stars"], []).append(r)
    for sv in sorted(by_star):
        rows = by_star[sv]
        n = len(rows)
        preds = {}
        for r in rows: preds[r["pred_class"]] = preds.get(r["pred_class"], 0) + 1
        mean_p1 = np.mean([r["p_onestar"] for r in rows])
        p(f"\n  Stars={sv}  n={n}  mean_P(one_star)={mean_p1:.4f}")
        for c in CLASS_NAMES:
            cnt = preds.get(c, 0)
            p(f"    → {c}: {cnt} ({100*cnt/n:.1f}%)")

    s45 = [r for r in all_img if r["stars"] in (4,5)]
    s1 = [r for r in all_img if r["stars"] == 1]
    if s45:
        fp = sum(1 for r in s45 if r["pred_class"]=="one_star")
        p(f"\n  SAFETY  4/5★ → one_star: {fp}/{len(s45)} = {100*fp/len(s45):.2f}%")
    if s1:
        tp = sum(1 for r in s1 if r["pred_class"]=="one_star")
        p(f"  DETECT  1★ → one_star:   {tp}/{len(s1)} = {100*tp/len(s1):.2f}%")

    # B) cluster-level
    p("\n" + "─" * 70)
    p("B) CLUSTER-LEVEL SEPARATION")
    p("─" * 70)
    junk = [c for c in all_clust if c["is_junk"]]
    nonjunk = [c for c in all_clust if not c["is_junk"]]
    p(f"  Junk: {len(junk)}   Non-junk: {len(nonjunk)}")

    from sklearn.metrics import roc_auc_score
    for m in ["max_p_onestar", "mean_p_onestar", "frac_pred_onestar"]:
        jv = [c[m] for c in junk]
        nv = [c[m] for c in nonjunk]
        if jv and nv:
            auc = roc_auc_score([1]*len(jv)+[0]*len(nv), jv+nv)
            p(f"\n  {m}:")
            p(f"    Junk     mean={np.mean(jv):.4f}  med={np.median(jv):.4f}  p25={np.percentile(jv,25):.4f}  p75={np.percentile(jv,75):.4f}")
            p(f"    Non-junk mean={np.mean(nv):.4f}  med={np.median(nv):.4f}  p25={np.percentile(nv,25):.4f}  p75={np.percentile(nv,75):.4f}")
            p(f"    AUC: {auc:.4f}")

    # C) threshold sweep
    p("\n" + "─" * 70)
    p("C) THRESHOLD SWEEP")
    p("─" * 70)
    for m in ["mean_p_onestar", "max_p_onestar", "frac_pred_onestar"]:
        p(f"\n  metric: {m}")
        p(f"  {'thr':>6} {'junk_caught':>11} {'recall':>7} {'nj_FP':>7} {'prec':>7} {'flagged':>8}")
        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            jc = sum(1 for c in junk if c[m] >= thr)
            nf = sum(1 for c in nonjunk if c[m] >= thr)
            tot = jc + nf
            pr = jc/tot if tot else 0
            rc = jc/len(junk) if junk else 0
            p(f"  {thr:>6.1f} {jc:>11} {rc:>7.3f} {nf:>7} {pr:>7.3f} {tot:>8}")

    # D) safety on has_45
    h45 = [c for c in all_clust if c["has_45"]]
    if h45:
        p(f"\n  SAFETY: {len(h45)} clusters have ≥1 4/5★ image")
        for thr in [0.3, 0.5, 0.7]:
            fl = sum(1 for c in h45 if c["mean_p_onestar"] >= thr)
            p(f"    thr={thr}: {fl}/{len(h45)} falsely flagged ({100*fl/len(h45):.1f}%)")

    with open(OUTPUT_DIR / "sample_results.txt", "w") as f:
        f.write("\n".join(out))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
