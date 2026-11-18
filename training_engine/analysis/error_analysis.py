"""Phase 5: Cross-model error analysis.

Joins predictions from best models on exam ID, analyzes agreement/disagreement
patterns, identifies hard negatives and complementary signals.

Models compared:
- nnU-Net segmentation (759 exams, n_positive_voxels as continuous score;
  c90z_384 input by default, --seg-dir to compare another run)
- MIP top90 k-fold (645 exams, 5-fold val predictions with probabilities)
- MIP center-crop k-fold (645 exams, 5-fold val predictions with probabilities)
- 3D R3D18K best (129 exams, single fold — included where available)

This is a one-off Phase 5 script, not a reusable analysis tool: the
experiment names/paths above are hardcoded to that specific run's outputs
(see build_joined_table's load_classification_predictions calls). Point it
at different experiments by editing those calls directly.

Usage:
    python -m analysis.error_analysis [--seg-dir ../segmentation/evaluation]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dlorchestra.analysis.predictions import load_fold_predictions

from analysis.bootstrap_auc import boot_ci, paired_boot_diff
from analysis.common import extract_exam_id

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
# c90z_384 is the best nnU-Net input (AUC 0.726); the Sep-2026 Phase 5 run
# mistakenly used "evaluation" (original NIfTIs, AUC 0.695).
SEGMENTATION_DIR = Path(__file__).parent.parent.parent / "segmentation" / "evaluation_c90z_384"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "phase5_error_analysis"


def load_classification_predictions(experiment_name: str) -> pd.DataFrame:
    """Load and concatenate predictions from all folds of an experiment."""
    combined = load_fold_predictions(EXPERIMENTS_DIR / experiment_name, id_fn=extract_exam_id)
    combined = combined.rename(columns={"id": "exam_id"})
    return combined[["exam_id", "true_label", "prob", "fold"]].copy()


def load_segmentation_predictions() -> pd.DataFrame:
    """Load nnU-Net segmentation predictions."""
    df = pd.read_csv(SEGMENTATION_DIR / "predictions.csv")
    # Use log1p(n_positive_voxels) as a continuous score — more spread than raw counts
    df["seg_score"] = np.log1p(df["n_positive_voxels"])
    # Normalize to [0, 1] range for comparability
    max_score = df["seg_score"].max()
    if max_score > 0:
        df["seg_score"] = df["seg_score"] / max_score
    return df[["exam_id", "true_label", "pred_label", "n_positive_voxels", "seg_score"]].copy()


def build_joined_table() -> pd.DataFrame:
    """Join all model predictions on exam_id."""
    print("Loading predictions...")

    # Segmentation (all 759)
    seg = load_segmentation_predictions()
    seg = seg.rename(columns={
        "pred_label": "seg_pred",
        "seg_score": "seg_prob",
    })

    # MIP top90 k-fold (645)
    mip_top90 = load_classification_predictions("MIP_earlyfusion_top90_kfold")
    mip_top90 = mip_top90.rename(columns={"prob": "mip_top90_prob"})
    mip_top90 = mip_top90.drop(columns=["fold"])

    # MIP center-crop k-fold (645)
    mip_center = load_classification_predictions("MIP_earlyfusion_mixed_v2_kfold")
    mip_center = mip_center.rename(columns={"prob": "mip_center_prob"})
    mip_center = mip_center.drop(columns=["true_label", "fold"])

    # 3D R3D18K best (129)
    r3d = load_classification_predictions("R3D18K_128_oversample_5515")
    r3d = r3d.rename(columns={"prob": "r3d_prob"})
    r3d = r3d.drop(columns=["true_label", "fold"])

    # Join — start from segmentation (full coverage), left join others
    joined = seg.merge(mip_top90, on="exam_id", how="left", suffixes=("", "_mip"))
    joined = joined.merge(mip_center, on="exam_id", how="left")
    joined = joined.merge(r3d, on="exam_id", how="left")

    # Use true_label from segmentation (most complete)
    joined = joined.drop(columns=["true_label_mip"], errors="ignore")

    print(f"  Segmentation: {seg.shape[0]} exams")
    print(f"  MIP top90 k-fold: {mip_top90.shape[0]} exams")
    print(f"  MIP center-crop k-fold: {mip_center.shape[0]} exams")
    print(f"  3D R3D18K: {r3d.shape[0]} exams")
    print(f"  Joined table: {joined.shape[0]} exams ({joined['mip_top90_prob'].notna().sum()} with MIP)")

    return joined


def per_model_auc(joined: pd.DataFrame):
    """Print per-model AUC on the common subset."""
    # Common subset where all three main models have predictions
    common = joined.dropna(subset=["mip_top90_prob", "mip_center_prob"])
    print(f"\n--- Per-model AUC (n={len(common)}, {common['true_label'].sum()} positive) ---")

    models = {
        "nnU-Net segmentation": "seg_prob",
        "MIP top90 (k-fold)": "mip_top90_prob",
        "MIP center-crop (k-fold)": "mip_center_prob",
    }
    for name, col in models.items():
        auc = roc_auc_score(common["true_label"], common[col])
        print(f"  {name}: AUC = {auc:.4f}")

    # Also on subset with 3D
    r3d_subset = joined.dropna(subset=["r3d_prob"])
    if len(r3d_subset) > 0 and r3d_subset["true_label"].sum() > 0:
        auc_r3d = roc_auc_score(r3d_subset["true_label"], r3d_subset["r3d_prob"])
        print(f"  3D R3D18K (n={len(r3d_subset)}): AUC = {auc_r3d:.4f}")


def threshold_predictions(joined: pd.DataFrame, seg_thresh: float = 0.5) -> pd.DataFrame:
    """Add binary predictions at threshold=0.5 for classification models, custom for segmentation."""
    df = joined.copy()
    # For segmentation, use the optimal threshold from prior evaluation
    # But also provide 0.5 on normalized score
    df["seg_binary"] = (df["seg_prob"] >= seg_thresh).astype(int)
    df["mip_top90_binary"] = (df["mip_top90_prob"] >= 0.5).astype(int)
    df["mip_center_binary"] = (df["mip_center_prob"] >= 0.5).astype(int)
    df["r3d_binary"] = (df["r3d_prob"] >= 0.5).astype(int)
    return df


def agreement_analysis(joined: pd.DataFrame):
    """Analyze agreement patterns across models."""
    # Focus on common subset (seg + both MIPs)
    common = joined.dropna(subset=["mip_top90_prob", "mip_center_prob"]).copy()
    common = threshold_predictions(common)

    positives = common[common["true_label"] == 1]
    negatives = common[common["true_label"] == 0]

    print(f"\n--- Agreement analysis (n={len(common)}: {len(positives)} pos, {len(negatives)} neg) ---")

    model_cols = ["seg_binary", "mip_top90_binary", "mip_center_binary"]
    model_names = ["nnU-Net", "MIP-top90", "MIP-center"]

    n_models = len(model_cols)

    # For positives: how many models detect each?
    pos_votes = positives[model_cols].sum(axis=1)
    print(f"\n  Positive cases — models agreeing on positive:")
    for n in range(n_models + 1):
        count = (pos_votes == n).sum()
        pct = count / len(positives) * 100 if len(positives) > 0 else 0
        print(f"    {n}/{n_models} models: {count} ({pct:.1f}%)")

    # Hard negatives: true positive, NO model detects
    hard_neg = positives[pos_votes == 0]
    print(f"\n  HARD NEGATIVES (aneurysm present, all {n_models} models miss): {len(hard_neg)}")
    if len(hard_neg) > 0:
        print(f"    Exam IDs: {', '.join(sorted(hard_neg['exam_id'].tolist()))}")

    # For negatives: false positive patterns
    neg_votes = negatives[model_cols].sum(axis=1)
    print(f"\n  Negative cases — false positive votes:")
    for n in range(n_models + 1):
        count = (neg_votes == n).sum()
        pct = count / len(negatives) * 100 if len(negatives) > 0 else 0
        print(f"    {n}/{n_models} models: {count} ({pct:.1f}%)")

    return common


def complementarity_analysis(common: pd.DataFrame):
    """Find cases where one model catches what another misses (complementary signal)."""
    positives = common[common["true_label"] == 1].copy()

    model_cols = ["seg_binary", "mip_top90_binary", "mip_center_binary"]
    model_names = ["nnU-Net", "MIP-top90", "MIP-center"]

    print(f"\n--- Complementarity (among {len(positives)} positive cases) ---")

    # Pairwise: A catches, B misses
    for i, (col_a, name_a) in enumerate(zip(model_cols, model_names)):
        for j, (col_b, name_b) in enumerate(zip(model_cols, model_names)):
            if i == j:
                continue
            a_only = positives[(positives[col_a] == 1) & (positives[col_b] == 0)]
            print(f"  {name_a} catches, {name_b} misses: {len(a_only)}")

    # Union sensitivity: at least one model detects
    any_detect = positives[model_cols].max(axis=1)
    union_sens = any_detect.mean()
    print(f"\n  Individual sensitivities (at threshold=0.5):")
    for col, name in zip(model_cols, model_names):
        sens = positives[col].mean()
        print(f"    {name}: {sens:.3f}")
    print(f"  Union (any model detects): {union_sens:.3f}")

    # Intersection: all models agree positive
    all_detect = positives[model_cols].min(axis=1)
    intersection_sens = all_detect.mean()
    print(f"  Intersection (all models agree): {intersection_sens:.3f}")


def ensemble_analysis(joined: pd.DataFrame):
    """Test simple ensemble strategies."""
    common = joined.dropna(subset=["mip_top90_prob", "mip_center_prob"]).copy()

    print(f"\n--- Ensemble AUC (n={len(common)}) ---")

    # Average of normalized scores
    common["avg_2mip"] = (common["mip_top90_prob"] + common["mip_center_prob"]) / 2
    common["avg_3model"] = (common["seg_prob"] + common["mip_top90_prob"] + common["mip_center_prob"]) / 3
    # Max
    common["max_3model"] = common[["seg_prob", "mip_top90_prob", "mip_center_prob"]].max(axis=1)

    ensembles = {
        "Avg(MIP-top90, MIP-center)": "avg_2mip",
        "Avg(nnU-Net, MIP-top90, MIP-center)": "avg_3model",
        "Max(nnU-Net, MIP-top90, MIP-center)": "max_3model",
    }

    y = common["true_label"].to_numpy()
    for name, col in ensembles.items():
        auc = roc_auc_score(y, common[col])
        lo, hi = boot_ci(y, common[col].to_numpy())
        print(f"  {name}: AUC = {auc:.4f} [{lo:.3f}, {hi:.3f}]")

    print("\n  Paired bootstrap, Avg(3) minus each member:")
    for name, col in [("nnU-Net", "seg_prob"), ("MIP-top90", "mip_top90_prob"), ("MIP-center", "mip_center_prob")]:
        d, lo, hi = paired_boot_diff(y, common["avg_3model"].to_numpy(), common[col].to_numpy())
        print(f"    vs {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    return common


def plot_probability_scatter(common: pd.DataFrame, output_dir: Path):
    """Scatter plots of model probabilities colored by true label."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        ("seg_prob", "mip_top90_prob", "nnU-Net score", "MIP top90 prob"),
        ("seg_prob", "mip_center_prob", "nnU-Net score", "MIP center prob"),
        ("mip_top90_prob", "mip_center_prob", "MIP top90 prob", "MIP center prob"),
    ]

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, pairs):
        subset = common.dropna(subset=[x_col, y_col])
        neg = subset[subset["true_label"] == 0]
        pos = subset[subset["true_label"] == 1]
        ax.scatter(neg[x_col], neg[y_col], c="tab:blue", alpha=0.3, s=15, label=f"Negative (n={len(neg)})")
        ax.scatter(pos[x_col], pos[y_col], c="tab:red", alpha=0.6, s=25, label=f"Positive (n={len(pos)})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Cross-model probability scatter (Phase 5 error analysis)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "probability_scatter.png", dpi=150)
    print(f"\n  Saved: {output_dir / 'probability_scatter.png'}")
    plt.close(fig)


def plot_detection_heatmap(common: pd.DataFrame, output_dir: Path):
    """Heatmap showing which models detect each positive case."""
    positives = common[common["true_label"] == 1].copy()
    positives = threshold_predictions(positives)

    model_cols = ["seg_binary", "mip_top90_binary", "mip_center_binary"]
    model_names = ["nnU-Net", "MIP-top90", "MIP-center"]

    # Sort by total votes (hardest cases first)
    positives["total_votes"] = positives[model_cols].sum(axis=1)
    positives = positives.sort_values(["total_votes", "exam_id"])

    matrix = positives[model_cols].values.T  # models × cases

    fig, ax = plt.subplots(figsize=(max(12, len(positives) * 0.15), 3))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xticks(range(len(positives)))
    ax.set_xticklabels(positives["exam_id"].values, rotation=90, fontsize=6)
    ax.set_title("Detection matrix — positive cases (green=detected, red=missed)")
    fig.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "detection_heatmap.png", dpi=150)
    print(f"  Saved: {output_dir / 'detection_heatmap.png'}")
    plt.close(fig)


def save_joined_table(joined: pd.DataFrame, output_dir: Path):
    """Save full joined predictions table for further analysis."""
    out_path = output_dir / "joined_predictions.csv"
    joined.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def main():
    global SEGMENTATION_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seg-dir", type=Path, default=SEGMENTATION_DIR,
                        help="nnU-Net evaluation dir with predictions.csv")
    SEGMENTATION_DIR = parser.parse_args().seg_dir
    print(f"nnU-Net predictions: {SEGMENTATION_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joined = build_joined_table()
    save_joined_table(joined, OUTPUT_DIR)
    per_model_auc(joined)

    common = agreement_analysis(joined)
    complementarity_analysis(common)
    ensemble_common = ensemble_analysis(joined)

    plot_probability_scatter(ensemble_common, OUTPUT_DIR)
    plot_detection_heatmap(common, OUTPUT_DIR)

    print(f"\nPhase 5 error analysis complete. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
