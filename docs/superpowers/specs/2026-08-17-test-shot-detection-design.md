# Test-shot detection — design spec

Date: 2026-08-17

## Problem

Photographers fire test shots before a real session/pose to check ISO/WB/shutter settings. We want a single-image probability score, `p(is_test_shot)`, computable per image with **no access to neighboring/burst images** (confirmed constraint — the detector sees one isolated image at a time).

Inspection of `test_shot_images/{Set1..Set14,others_akshay_found}/test/` (36 labeled test shots, 54 labeled-normal siblings, plus Set8/Set9 which contain zero test shots and serve as pure hard negatives) surfaced three distinct visual failure modes:

1. **Exposure-test frames** (majority of examples): same pose/composition as the real shot, but exposure wasn't dialed in yet — near-black or near-white, essentially no local detail (Sobel edge energy 1–8 vs. 20–60+ for real photos, including legitimately dark/moody ones).
2. **Junk/wrong-subject frames** (Set1, Set11): camera fired at an incidental nearby object (a stone pillar, a diamond-plate truck bed) while the photographer adjusted settings — normal exposure, but no face/person and generic repetitive texture instead of the actual scene.
3. **Near-duplicate frames** (Set3's 79620/79621, and Set2's 79635/79640 once the directional-artifact reading is excluded — see v2 note below): visually equivalent to the kept shot by every pixel statistic tried (histogram, edge energy, per-channel color ratios). These are "test shots" only because the photographer picked a different frame moments later, not because of any visual defect. **No single-image signal can recover these — confirmed empirically** (see Validation below); they are out of scope for this detector and should not be chased.

## Design: two independent rule-based scores, no ML training

With ~40 positives spread across 14 shoots, training a classifier from scratch would memorize shoot-specific backgrounds rather than learn anything general. Both scores below are hand-tuned, deterministic, and inspectable.

### Score A — exposure clipping (catches failure mode 1)

On a downsampled grayscale copy:
- `frac_dark` = fraction of pixels below ~10, or `frac_bright` = fraction above ~245 (histogram clipping)
- `edge_energy` = mean Sobel gradient magnitude (local detail/information measure)

Clipping fraction alone is not sufficient — Set8/Set9's legitimately dark real photos reach `frac_dark` up to 43% while keeping real edge structure (Sobel 25–55). Test shots combine high clipping **with** near-zero edge energy (Sobel 1–8). Both conditions must hold jointly.

A hard clipping fraction alone also *misses* real cases: Set3's 79622 (mean luminance ≈19, clearly a bad dark frame) only hits 15% `frac_dark` because most of its pixels sit at 12–26 — dark enough to look bad, not below the strict clipping cutoff. Adding a continuous low-mean-luminance term alongside the hard clipping fraction (both still gated by the same low-detail requirement) catches this without introducing any new false positives on Set8/Set9's legitimately dim-but-detailed photos:

```
info_score    = clamp(edge_energy / EDGE_REF, 0, 1)          # EDGE_REF ≈ 30, a typical detailed-photo edge energy
clip_score    = max(frac_dark, frac_bright)
low_light     = 1 - clamp(mean_luminance / MEAN_REF, 0, 1)   # MEAN_REF ≈ 70
combined_dark = max(clip_score, low_light)
score_A       = combined_dark * (1 - info_score)
```

### Score B — no-subject / generic content (catches failure mode 2)

Combines three inputs, with face/person-detection confirmed available in the pipeline at full resolution (so missed small/distant faces should be rare):

- `no_subject` = true if the face/person detector finds nothing
- `texture_repetitiveness` = local self-similarity / autocorrelation of the gradient field (high for repetitive/generic surfaces like stone or diamond plate, low for real scenes with varied structure)
- `bank_similarity` = mobilenet-embedding cosine similarity to a small reference bank built from the known junk examples (Set1/9500, Set11's four images) — a minor, expandable term, not the primary driver

`no_subject` alone is **not sufficient** — legitimate detail shots (rings, flowers, decor) also have no face and must not be flagged. The texture-repetitiveness term is the real gate; face-absence only adds confidence when both fire together:

```
score_B = texture_repetitiveness * (no_subject ? 1.0 : 0.4)   [ * blend_with(bank_similarity, weight=minor) ]
```

A working prototype of `texture_repetitiveness` (8×8 block grid; `macro_uniformity = 1 - clamp(block_mean_std / BLOCK_STD_REF, 0, 1)` combined with `texture_richness = clamp(edge_energy / EDGE_REF2, 0, 1)`) was implemented and measured against the labeled set (Score B alone: AUC 0.887). It has one known gap worth flagging before implementation treats this as settled: **Set11's rock-texture close-ups — the primary motivating example for this score — score too low** (0.00–0.08, well under any reasonable threshold), because the macro shading across that particular close-up isn't as spatially flat as the block-uniformity formula assumes. This needs real iteration on the texture measure itself during implementation, not just threshold tuning. `bank_similarity` (mobilenet-embedding similarity to a reference bank) was not implemented in this validation pass; it remains the minor/expandable term described above and could plausibly help specifically with this Set11 gap.

### Combination

```
p(is_test_shot) = max(score_A, score_B)
```

Either failure mode alone is sufficient evidence; each score is mapped through a calibration curve fit against the labeled set.

### Validated results (this labeled set, threshold 0.3)

| | AUC |
|---|---|
| Score A alone | 0.902 |
| Score B alone | 0.887 |
| **max(A, B) combined** | **0.941** |

At `threshold = 0.3`: **0/54 false positives** on normal images, **27/36 (75%) recall** on test shots. The 9 misses:

- **Set11, all 4** — the texture-formula gap above (open item, not out-of-scope).
- **Set2 (79635, 79640), Set3 (79620, 79621)** — the near-duplicate/no-defect cases from failure mode 3, confirmed out of scope (see below).
- **Set7 (J75_6303)** — one borderline partial-exposure case, tunable with a lower threshold at some FP-risk cost.

This compares favorably to the existing TinyViT visual-rejection model (AUC 0.755, 35% false-positive rate at its natural decision boundary — see below).

## Auxiliary signal considered and rejected as a primary input

`src/visual_rejection_sample.py`'s existing TinyViT 3-class model (`unrated`/`one_star`/`four_five`, trained on photographer star ratings) was tested directly against this dataset:

- `AUC = 0.725` using `P(one_star)`, `AUC = 0.755` using the raw logit gap `z(one_star) - z(four_five)` — a small improvement, not a fix, since it's measuring the same underlying signal.
- **19/54 (35%) false-positive rate** on normal images if used as a standalone reject flag — entirely driven by Sets 4–7, where *100% of legitimate keepers* score as high or higher than the real test shots sitting next to them. The model has learned "dim/dark evening scene → reject-leaning," which dominates over anything specific to calibration frames.
- **4/36 (11%) false negatives** — exactly the near-duplicate/no-defect cases from failure mode 3 (79635, 79640, 79620, 79621), confirming it has no signal to offer there either; flipping the sign (`z45 - z1`) is mathematically the same information reversed and does not help.

Conclusion: not safe as a standalone or primary signal. May be folded in later as a minor auxiliary/tie-breaking term, gated behind Score A/B so it can't override them, but it is **not part of this v1 design**.

## Deferred to v2: directional exposure-ramp signal (flash-sync / shutter-curtain artifact)

A third failure signature was identified during review: 79635/79640/79620 show a steep, monotonic left-to-right brightness collapse (5-strip profile ~190→180→145→105→~25, darkest/brightest ratio ≈ 0.13–0.14) consistent with a focal-plane-shutter/flash-sync-speed miss — i.e. an actual shutter-speed calibration defect, directly matching the original ISO/WB/shutter framing. Normal siblings in the same sets show ratios of 0.82–1.08 (flat or mild natural vignette).

However, a naive whole-frame version of this metric is **not usable as-is**: tested across the full dataset, it produces 18/54 false positives on normal images, because venue/event photos have real spatial brightness variation (a subject or window to one side) that a coarse directional average cannot distinguish from a curtain artifact. It only read cleanly on Set2/Set3 because those are flat-studio-backdrop sessions where a large, near-uniform background fills most of the frame — the only condition under which a directional-brightness metric is unambiguous.

**v2 direction** (not built now): restrict the ramp measurement to detected flat/low-texture background regions only (reusing the same low-local-variance detection Score A/B already need), rather than the whole frame. This should isolate genuine curtain-sync artifacts from ordinary foreground-driven lighting variation. Needs more flat-backdrop examples of this specific artifact to calibrate before it's trustworthy — deferred until then.

## Validation plan

- **Leave-one-set-out**: fit thresholds/parameters on 13 sets, score the held-out set, repeat for all 14. Report per-set recall/precision to get an honest read on generalization rather than overfitting to these shoots.
- **Dedicated false-positive check on Set8/Set9**: these are pure-negative sets (zero test shots). Report their false-positive rate as its own line item, separate from aggregate precision — a single false positive there is a clearer signal of a bad threshold than an aggregate number would show.
- **Explicit known-miss list**: Set3's 79620/79621 (and, pending the v2 ramp signal, Set2's 79635/79640) are expected false negatives in v1 and should be reported as such, not silently absorbed into a recall number.

## Deliverable

A standalone Python script, sibling to `generate_mobilenet_embeddings.py`, taking an image path (or folder) and outputting `p(is_test_shot)` plus which score fired, along with the validation report against this labeled dataset.
