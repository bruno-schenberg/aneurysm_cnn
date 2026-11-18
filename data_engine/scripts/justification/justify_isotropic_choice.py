"""
Justify the isotropic resampling design choice.

Compares three Z-axis strategies for building fixed-size 3D volumes:

  A) Isotropic (chosen): uniform spacing in all axes = 180mm / dim.
     All anatomy fits. ~26% Z padding (air). Simple, rotation-invariant.

  B) Shrink Z to target: resample Z so all anatomy fits in exactly
     128 or 256 slices. Finer Z than iso (~1.04 vs 1.41 mm for 128),
     but variable Z spacing across exams and anisotropic voxels break
     rotation augmentations.

  C) 1:1 native Z: keep native Z spacing, take 128/256 centre slices.
     Finest Z resolution, but crops ~47% of anatomy for 128 target.
     For 256 target, only ~5% cropped on average, but a 256-slice
     non-iso variant can't pair with a 128-slice non-iso variant
     (the 128 version crops too much).

Conclusion: isotropic wins because (1) no anatomy is cropped, (2) uniform
voxel size across the dataset, (3) augmentations (rotation, flip) are
valid in all axes. The ~1.36x coarser Z vs shrink-to-target is acceptable
given the 180mm FOV and typical aneurysm sizes (3-25mm).

The old 8-variant plan (crop/shrink x iso/noniso x 128/256) was dropped
after centroid crop standardised XY FOV to 180mm, making isotropic
resampling straightforward and reducing the design space to 2 variants.

Usage:
    python scripts/justification/justify_isotropic_choice.py

Outputs:
    reports/isotropic_justification.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

TARGET_FOV = 180.0  # mm, after centroid crop


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.DATASETS_REPORT)
    df = df[df["variant"] == "iso_128"].copy()
    df["input_z"] = df["input_shape"].str.split("x").str[2].astype(int)
    df["anatomy_z_mm"] = TARGET_FOV * (1 - df["pct_pad"] / 100)
    df["native_z_approx"] = df["anatomy_z_mm"] / df["input_z"]
    return df


def main() -> None:
    df = load_data()
    n = len(df)

    iso_z_128 = TARGET_FOV / 128
    iso_z_256 = TARGET_FOV / 256

    # --- Strategy B: shrink Z to target ---
    df["shrink_z_128"] = df["anatomy_z_mm"] / 128
    df["shrink_z_256"] = df["anatomy_z_mm"] / 256

    # --- Strategy C: 1:1 native Z, centre-crop to target ---
    df["native_cover_128"] = df["native_z_approx"] * 128
    df["native_crop_128_pct"] = np.clip(
        1 - df["native_cover_128"] / df["anatomy_z_mm"], 0, 1
    ) * 100
    df["native_cover_256"] = df["native_z_approx"] * 256
    df["native_crop_256_pct"] = np.clip(
        1 - df["native_cover_256"] / df["anatomy_z_mm"], 0, 1
    ) * 100

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Isotropic vs alternative Z strategies ({n} exams, {TARGET_FOV:.0f} mm FOV)",
        fontsize=14, fontweight="bold",
    )

    # Row 0: Z spacing comparison for 128 and 256 targets
    # Col 0: Z spacing distributions for 128 target
    ax = axes[0, 0]
    bins = np.linspace(0.2, 1.6, 50)
    ax.hist(df["native_z_approx"], bins=bins, alpha=0.6, label="C: native Z", color="tab:green")
    ax.hist(df["shrink_z_128"], bins=bins, alpha=0.6, label="B: shrink-to-128", color="tab:orange")
    ax.axvline(iso_z_128, color="tab:blue", lw=2, ls="--", label=f"A: iso ({iso_z_128:.3f} mm)")
    ax.set_xlabel("Z spacing (mm)")
    ax.set_ylabel("Exam count")
    ax.set_title("Z spacing — 128 target")
    ax.legend(fontsize=8)

    # Col 1: Z spacing distributions for 256 target
    ax = axes[0, 1]
    bins = np.linspace(0.2, 0.9, 50)
    ax.hist(df["native_z_approx"], bins=bins, alpha=0.6, label="C: native Z", color="tab:green")
    ax.hist(df["shrink_z_256"], bins=bins, alpha=0.6, label="B: shrink-to-256", color="tab:orange")
    ax.axvline(iso_z_256, color="tab:blue", lw=2, ls="--", label=f"A: iso ({iso_z_256:.4f} mm)")
    ax.set_xlabel("Z spacing (mm)")
    ax.set_ylabel("Exam count")
    ax.set_title("Z spacing — 256 target")
    ax.legend(fontsize=8)

    # Col 2: Coarsening ratio (iso / shrink) for 128
    ax = axes[0, 2]
    ratio = iso_z_128 / df["shrink_z_128"]
    ax.hist(ratio, bins=40, color="tab:red", alpha=0.7)
    ax.axvline(ratio.median(), color="black", lw=2, ls="--",
               label=f"median: {ratio.median():.2f}x")
    ax.set_xlabel("Iso Z / shrink-to-128 Z (coarsening factor)")
    ax.set_ylabel("Exam count")
    ax.set_title("How much coarser is iso vs shrink? (128)")
    ax.legend(fontsize=8)

    # Row 1: Anatomy cropping and padding
    # Col 0: Anatomy Z extent + padding in iso 128
    ax = axes[1, 0]
    ax.hist(df["anatomy_z_mm"], bins=40, color="tab:blue", alpha=0.7, label="Anatomy Z (mm)")
    ax.axvline(TARGET_FOV, color="red", lw=2, ls="--", label=f"Cube Z = {TARGET_FOV:.0f} mm")
    ax.axvline(df["anatomy_z_mm"].median(), color="black", lw=1.5, ls=":",
               label=f"Median: {df['anatomy_z_mm'].median():.0f} mm")
    ax.set_xlabel("Z extent (mm)")
    ax.set_ylabel("Exam count")
    ax.set_title("A (iso): anatomy vs cube — no crop, ~26% pad")
    ax.legend(fontsize=8)

    # Col 1: Anatomy cropped by 1:1 native Z at 128 slices
    ax = axes[1, 1]
    ax.hist(df["native_crop_128_pct"], bins=40, color="tab:green", alpha=0.7)
    ax.axvline(df["native_crop_128_pct"].median(), color="black", lw=2, ls="--",
               label=f"Median: {df['native_crop_128_pct'].median():.0f}% cropped")
    ax.set_xlabel("Anatomy cropped (%)")
    ax.set_ylabel("Exam count")
    ax.set_title("C (1:1 native): anatomy LOST at 128 slices")
    ax.legend(fontsize=8)

    # Col 2: Anatomy cropped by 1:1 native Z at 256 slices
    ax = axes[1, 2]
    ax.hist(df["native_crop_256_pct"], bins=40, color="tab:green", alpha=0.7)
    ax.axvline(df["native_crop_256_pct"].median(), color="black", lw=2, ls="--",
               label=f"Median: {df['native_crop_256_pct'].median():.0f}% cropped")
    ax.set_xlabel("Anatomy cropped (%)")
    ax.set_ylabel("Exam count")
    ax.set_title("C (1:1 native): anatomy LOST at 256 slices")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = config.REPORTS_DIR / "isotropic_justification.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Console summary ---
    print(f"Saved {out_path}")
    print()
    print(f"{'Strategy':<30s}  {'Z spacing (mm)':<18s}  {'Anatomy cropped':<18s}  Notes")
    print("-" * 100)
    print(f"{'A: Isotropic (chosen)':<30s}  "
          f"{'128: %.3f / 256: %.4f' % (iso_z_128, iso_z_256):<18s}  "
          f"{'0%':<18s}  ~26% Z padding (air)")
    print(f"{'B: Shrink Z to target':<30s}  "
          f"{'128: %.3f / 256: %.4f' % (df['shrink_z_128'].median(), df['shrink_z_256'].median()):<18s}  "
          f"{'0%':<18s}  Variable Z spacing per exam")
    print(f"{'C: 1:1 native Z':<30s}  "
          f"{'%.3f (median native)' % df['native_z_approx'].median():<18s}  "
          f"{'128: %d%% / 256: %d%%' % (df['native_crop_128_pct'].median(), df['native_crop_256_pct'].median()):<18s}  "
          f"Finest Z but crops anatomy")
    print()
    print(f"Iso 128 is {(iso_z_128 / df['shrink_z_128'].median()):.2f}x coarser in Z than shrink-to-128")
    print(f"Iso 256 is {(iso_z_256 / df['shrink_z_256'].median()):.2f}x coarser in Z than shrink-to-256")
    print()
    print("1:1 native Z at 128 slices covers only ~64mm (median) of ~136mm anatomy → ~50% lost")
    print("1:1 native Z at 256 slices covers ~128mm → viable for 256 but no matching 128 variant")


if __name__ == "__main__":
    main()
