"""Evaluate nnU-Net segmentation predictions as binary classification.

For each segmentation mask:
  - Any nonzero voxel -> predicted positive (class 1)
  - All zero -> predicted negative (class 0)

Uses voxel count as a continuous confidence score for AUC.

Usage:
    python evaluate.py /path/to/predictions mapping.csv [--output-dir results/]
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)


def load_predictions(pred_dir: str, mapping_csv: str) -> list[dict]:
    pred_path = Path(pred_dir)

    with open(mapping_csv) as f:
        mapping = list(csv.DictReader(f))

    results = []
    missing = 0
    for row in mapping:
        exam_id = row["exam_id"]
        label = int(row["label"])
        if label == -1:
            continue

        pred_file = pred_path / f"{exam_id}.nii.gz"
        if not pred_file.exists():
            missing += 1
            continue

        mask = nib.load(str(pred_file)).get_fdata()
        n_pos_voxels = int(np.count_nonzero(mask))

        results.append({
            "exam_id": exam_id,
            "true_label": label,
            "pred_label": int(n_pos_voxels > 0),
            "n_positive_voxels": n_pos_voxels,
        })

    if missing:
        print(f"WARNING: {missing} exams had no prediction file")

    return results


def evaluate(results: list[dict], output_dir: str | None = None):
    y_true = np.array([r["true_label"] for r in results])
    y_pred = np.array([r["pred_label"] for r in results])
    n_voxels = np.array([r["n_positive_voxels"] for r in results])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f2 = fbeta_score(y_true, y_pred, beta=2)
    auc = roc_auc_score(y_true, n_voxels) if len(np.unique(y_true)) > 1 else float("nan")

    print("=" * 60)
    print("Segmentation -> Classification Evaluation")
    print("=" * 60)
    print(f"Samples: {len(results)}  (positives: {int(y_true.sum())}, negatives: {int((1-y_true).sum())})")
    print()
    print(f"  TP: {tp:4d}   FP: {fp:4d}")
    print(f"  FN: {fn:4d}   TN: {tn:4d}")
    print()
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  F2:          {f2:.4f}")
    print(f"  AUC (voxel): {auc:.4f}")

    # Voxel count distribution
    print()
    print("Segmented voxel counts:")
    for cls, name in [(1, "True positives"), (0, "True negatives")]:
        mask_cls = y_true == cls
        if mask_cls.any():
            v = n_voxels[mask_cls]
            det = (v > 0).sum()
            print(f"  {name} (n={mask_cls.sum()}): detected={det}, "
                  f"median_voxels={np.median(v[v>0]):.0f}" if det > 0
                  else f"  {name} (n={mask_cls.sum()}): detected=0")

    # Threshold sweep on voxel count
    if len(np.unique(y_true)) > 1:
        print()
        print("Threshold sweep (min voxels to call positive):")
        print(f"  {'Threshold':>10s}  {'Sens':>6s}  {'Spec':>6s}  {'Prec':>6s}  {'F2':>6s}")
        for thresh in [1, 5, 10, 50, 100, 500]:
            y_t = (n_voxels >= thresh).astype(int)
            if y_t.sum() == 0 or y_t.sum() == len(y_t):
                continue
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_t, labels=[0, 1]).ravel()
            sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
            prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            f2_t = fbeta_score(y_true, y_t, beta=2)
            print(f"  {thresh:>10d}  {sens_t:>6.3f}  {spec_t:>6.3f}  {prec_t:>6.3f}  {f2_t:>6.3f}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "predictions.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        summary = {
            "n_samples": len(results),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f2": float(f2),
            "auc_voxel_count": float(auc),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation as classification")
    parser.add_argument("pred_dir", help="Directory with nnU-Net prediction NIfTI masks")
    parser.add_argument("mapping_csv", help="Mapping CSV from prepare_input.py")
    parser.add_argument("--output-dir", help="Save predictions.csv and summary.json")
    args = parser.parse_args()

    results = load_predictions(args.pred_dir, args.mapping_csv)
    if results:
        evaluate(results, args.output_dir)
    else:
        print("ERROR: No valid predictions found")
