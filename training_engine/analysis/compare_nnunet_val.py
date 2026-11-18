"""Compare nnU-Net predictions on the same val split used by MIP experiments.

Reads the MIP experiment's predictions.csv to identify which exam IDs were in
the val set, then filters the nnU-Net predictions to those same IDs and
computes AUC + other metrics.

Usage:
    python -m analysis.compare_nnunet_val \
        --mip-predictions experiments/MIP_earlyfusion_mixed_v2/fold_1/predictions.csv \
        --nnunet-predictions ../segmentation/evaluation/predictions.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score, roc_auc_score

from analysis.common import extract_exam_id


def main() -> None:
    script_dir = Path(__file__).resolve().parent.parent
    default_mip = script_dir / "experiments/MIP_earlyfusion_mixed_v2/fold_1/predictions.csv"
    default_nnunet = script_dir.parent / "segmentation/evaluation/predictions.csv"

    parser = argparse.ArgumentParser(description="Compare nnU-Net on MIP val split")
    parser.add_argument("--mip-predictions", default=str(default_mip),
                        help="Path to MIP experiment's predictions.csv (defines val set)")
    parser.add_argument("--nnunet-predictions", default=str(default_nnunet),
                        help="Path to nnU-Net predictions.csv (all 759 volumes)")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Min voxels to call positive for nnU-Net (default: 1)")
    args = parser.parse_args()

    # Read MIP val set exam IDs
    mip_exam_ids = set()
    mip_labels = {}
    mip_probs = {}
    with open(args.mip_predictions) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = extract_exam_id(row["image"])
            mip_exam_ids.add(eid)
            mip_labels[eid] = int(row["true_label"])
            mip_probs[eid] = float(row["prob"])

    # Read nnU-Net predictions
    nnunet_data = {}
    with open(args.nnunet_predictions) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["exam_id"]
            nnunet_data[eid] = {
                "true_label": int(row["true_label"]),
                "pred_label": int(row["pred_label"]),
                "n_positive_voxels": int(row["n_positive_voxels"]),
            }

    # Filter nnU-Net to MIP val set
    matched = mip_exam_ids & set(nnunet_data.keys())
    missing = mip_exam_ids - set(nnunet_data.keys())

    print(f"MIP val set: {len(mip_exam_ids)} exams")
    print(f"nnU-Net predictions: {len(nnunet_data)} exams")
    print(f"Matched: {len(matched)} exams")
    if missing:
        print(f"Missing from nnU-Net: {len(missing)} ({sorted(missing)[:5]}...)")

    # Compute metrics for both on the same samples
    exam_ids = sorted(matched)
    true_labels = np.array([mip_labels[eid] for eid in exam_ids])
    n_pos = true_labels.sum()
    n_neg = len(true_labels) - n_pos

    print(f"\nVal set composition: {len(exam_ids)} samples ({n_pos} positive, {n_neg} negative)")

    # MIP metrics
    mip_p = np.array([mip_probs[eid] for eid in exam_ids])
    mip_auc = roc_auc_score(true_labels, mip_p)
    mip_pred = (mip_p >= 0.5).astype(int)
    mip_tp = ((mip_pred == 1) & (true_labels == 1)).sum()
    mip_fp = ((mip_pred == 1) & (true_labels == 0)).sum()
    mip_fn = ((mip_pred == 0) & (true_labels == 1)).sum()
    mip_tn = ((mip_pred == 0) & (true_labels == 0)).sum()

    print(f"\n{'='*60}")
    print(f"MIP earlyfusion_mixed_v2 (on this val split)")
    print(f"{'='*60}")
    print(f"  AUC:         {mip_auc:.4f}")
    print(f"  Sensitivity: {mip_tp/(mip_tp+mip_fn):.3f}" if (mip_tp+mip_fn) > 0 else "  Sensitivity: N/A")
    print(f"  Specificity: {mip_tn/(mip_tn+mip_fp):.3f}" if (mip_tn+mip_fp) > 0 else "  Specificity: N/A")
    print(f"  TP={mip_tp}  FP={mip_fp}  FN={mip_fn}  TN={mip_tn}")

    # nnU-Net metrics at multiple thresholds
    nnunet_voxels = np.array([nnunet_data[eid]["n_positive_voxels"] for eid in exam_ids])

    print(f"\n{'='*60}")
    print(f"nnU-Net (on same val split)")
    print(f"{'='*60}")

    # AUC using voxel count as continuous score
    nnunet_auc = roc_auc_score(true_labels, nnunet_voxels)
    print(f"  AUC (voxel count as score): {nnunet_auc:.4f}")

    # Threshold sweep
    print(f"\n  {'Threshold':>10s}  {'Sens':>6s}  {'Spec':>6s}  {'Prec':>6s}  {'F2':>6s}  {'TP':>4s}  {'FP':>4s}")
    for thresh in [1, 5, 10, 25, 50, 100]:
        pred = (nnunet_voxels >= thresh).astype(int)
        tp = ((pred == 1) & (true_labels == 1)).sum()
        fp = ((pred == 1) & (true_labels == 0)).sum()
        fn = ((pred == 0) & (true_labels == 1)).sum()
        tn = ((pred == 0) & (true_labels == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f2 = fbeta_score(true_labels, pred, beta=2, pos_label=1, zero_division=0.0)
        print(f"  {thresh:>10d}  {sens:>6.3f}  {spec:>6.3f}  {prec:>6.3f}  {f2:>6.3f}  {tp:>4d}  {fp:>4d}")

    # Compare
    print(f"\n{'='*60}")
    print(f"Comparison on same {len(exam_ids)} val samples")
    print(f"{'='*60}")
    delta = mip_auc - nnunet_auc
    print(f"  MIP AUC:    {mip_auc:.4f}")
    print(f"  nnU-Net AUC: {nnunet_auc:.4f}")
    print(f"  Delta:       {delta:+.4f} ({'MIP better' if delta > 0 else 'nnU-Net better'})")


if __name__ == "__main__":
    main()
