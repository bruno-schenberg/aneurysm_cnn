"""
compute_optimal_spacing.py

Computes the weighted-optimal fixed isotropic voxel spacing for variant A
of each target shape, given the known FOV distribution of the dataset.

## Background

Variant A resamples every volume to a fixed isotropic spacing, then uses
ResizeWithPadOrCrop to fit the target grid. The spacing is "fixed" (not
per-exam) so that every voxel always represents the same physical volume
regardless of the source scanner -- this is variant A's scientific purpose.

The choice of spacing involves a trade-off:
  - Too fine  → resampled volume larger than target → content gets cropped
  - Too coarse → resampled volume smaller than target → grid is zero-padded

This script finds the spacing that maximises a weighted efficiency metric
across the real dataset distribution.

## Efficiency metric

For a given spacing s and exam FOV (Ex, Ey, Ez) mapped to target (Tx, Ty, Tz):

    resampled_i = FOV_i / s   (approximate; MONAI rounds to nearest int)

    axis_score_i = min(resampled_i, Ti) / max(resampled_i, Ti)

    efficiency = axis_score_x × axis_score_y × axis_score_z

This equals 1.0 on a perfect fit and decreases symmetrically for both crop
(content loss) and pad (wasted grid capacity). The weighted efficiency is
the exam-count-weighted mean across all representative groups.

## Dataset groups

Four scanner/protocol archetypes cover the full dataset (739 exams total):

  Group   FOV (mm)        Count   % of dataset
  BP002   180 × 180 × 142   322      43.6 %
  BP042   240 × 240 × 130   218      29.5 %
  BP082   163 × 180 × 136   183      24.8 %   (non-square XY)
  BP240   240 × 240 × 125    16       2.2 %

BP082 is non-square in XY (163 mm × 180 mm). Both axes are scored
independently against the square XY target dimension.

## Usage

    python data_engine/compute_optimal_spacing.py

No arguments required. Output is printed to stdout.
Re-run whenever the dataset distribution or target shapes change.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Dataset distribution
# Each entry: (FOV_x_mm, FOV_y_mm, FOV_z_mm, exam_count)
# ---------------------------------------------------------------------------
GROUPS = [
    (180, 180, 142, 322),   # BP002 — 0.2344 mm XY, 0.5 mm Z
    (240, 240, 130, 218),   # BP042 — 0.4688 mm XY, 0.6 mm Z
    (163, 180, 136, 183),   # BP082 — 0.2344 mm XY, 0.5 mm Z (non-square XY)
    (240, 240, 125,  16),   # BP240 — 0.4688 mm XY, 0.3 mm Z
]
TOTAL = sum(g[3] for g in GROUPS)

# ---------------------------------------------------------------------------
# Target shapes to evaluate
# ---------------------------------------------------------------------------
TARGETS = {
    "128x128x128": (128, 128, 128),
    "192x192x128": (192, 192, 128),
    "256x256x176": (256, 256, 176),
}

# ---------------------------------------------------------------------------
# Principled candidates to always report alongside the optimum
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("min_XY / Tx  (fills smallest common head XY)", lambda tx, ty, tz: 180 / tx),
    ("1.4062 mm  (180/128, current 128³ candidate)",  lambda tx, ty, tz: 1.4062),
    ("1.0000 mm  (round number)",                     lambda tx, ty, tz: 1.0),
    ("0.9375 mm  (old 256×256×128 spacing)",          lambda tx, ty, tz: 0.9375),
    ("0.9000 mm",                                     lambda tx, ty, tz: 0.9),
    ("0.7500 mm",                                     lambda tx, ty, tz: 0.75),
]


def axis_score(fov: float, s: float, t: int) -> float:
    """Efficiency score for one axis: 1.0 = perfect, <1.0 = crop or pad."""
    r = fov / s
    return min(r, t) / max(r, t)


def weighted_efficiency(s: float, tx: int, ty: int, tz: int) -> float:
    """Exam-count-weighted mean efficiency across all groups."""
    total_score = sum(
        cnt * axis_score(fx, s, tx) * axis_score(fy, s, ty) * axis_score(fz, s, tz)
        for fx, fy, fz, cnt in GROUPS
    )
    return total_score / TOTAL


def per_group_scores(s: float, tx: int, ty: int, tz: int) -> list[float]:
    return [
        axis_score(fx, s, tx) * axis_score(fy, s, ty) * axis_score(fz, s, tz)
        for fx, fy, fz, _ in GROUPS
    ]


def main() -> None:
    group_labels = ["BP002", "BP042", "BP082", "BP240"]

    for shape_key, (tx, ty, tz) in TARGETS.items():
        print(f"\n{'=' * 66}")
        print(f"Target shape: {shape_key}  ({tx} × {ty} × {tz} voxels)")
        print(f"{'=' * 66}")

        # Grid search over spacing values
        spacings = np.arange(0.40, 1.80, 0.005)
        scores = [(s, weighted_efficiency(s, tx, ty, tz)) for s in spacings]
        best_s, best_score = max(scores, key=lambda x: x[1])

        # Print neighbourhood around optimum
        print(f"\n  Grid search (neighbourhood around optimum):")
        print(f"  {'spacing':>10}  {'wt_eff':>8}  " +
              "  ".join(f"{lab:>6}" for lab in group_labels))
        for s, score in scores:
            if abs(s - best_s) <= 0.035:
                pg = per_group_scores(s, tx, ty, tz)
                marker = " ◀ optimum" if abs(s - best_s) < 0.003 else ""
                print(f"  s={s:.3f} mm  {score:.4f}  " +
                      "  ".join(f"{v:>6.3f}" for v in pg) + marker)

        # Print principled candidates
        print(f"\n  Principled candidates:")
        print(f"  {'label':42s}  {'spacing':>10}  {'wt_eff':>8}  " +
              "  ".join(f"{lab:>6}" for lab in group_labels))
        for label, s_fn in CANDIDATES:
            s = s_fn(tx, ty, tz)
            score = weighted_efficiency(s, tx, ty, tz)
            pg = per_group_scores(s, tx, ty, tz)
            print(f"  {label:42s}  s={s:.4f}  {score:.4f}  " +
                  "  ".join(f"{v:>6.3f}" for v in pg))

        # Recommendation
        # Use the principled "min_XY / Tx" spacing if it scores within 0.5%
        # of the grid-search optimum; otherwise use the grid optimum.
        principled_s = 180 / tx
        principled_score = weighted_efficiency(principled_s, tx, ty, tz)
        gap = best_score - principled_score
        if gap < 0.005:
            rec_s = principled_s
            rec_label = f"180 / {tx} = {principled_s:.4f} mm (principled, within {gap*100:.2f}% of optimum)"
        else:
            rec_s = best_s
            rec_label = f"{best_s:.3f} mm (grid search optimum)"
        print(f"\n  Recommendation: {rec_label}")
        print(f"  Coverage at recommended spacing:")
        for (fx, fy, fz, cnt), lab in zip(GROUPS, group_labels):
            rx = fx / rec_s
            ry = fy / rec_s
            rz = fz / rec_s
            def describe(r, t):
                if r > t + 0.5:
                    return f"crop {r-t:.0f} px ({(r-t)/r*100:.1f}%)"
                elif r < t - 0.5:
                    return f"pad  {t-r:.0f} px ({(t-r)/t*100:.1f}%)"
                else:
                    return "exact fit"
            print(f"    {lab} (n={cnt:3d}): "
                  f"X {describe(rx, tx)}, "
                  f"Y {describe(ry, ty)}, "
                  f"Z {describe(rz, tz)}")


if __name__ == "__main__":
    main()
