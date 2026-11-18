"""Compare ensemble member subsets on the dev OOF only (nested CV); never reads test labels.

Run before analysis.ensemble to decide its member list. Also checks
age as an extra member and the nnU-Net input choice (c90z_384 vs original) on dev exams.

Usage (from training_engine/): python -m analysis.ensemble_subsets
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.bootstrap_auc import paired_boot_diff
from analysis.ensemble import fit_rule, load_member, load_nnunet
from analysis.splits import label_only_folds
from config import DATASET_CSV

SEG = Path("../segmentation")
MEMBERS = {"2D": "experiments/2D-3_512", "3D_aniso": "experiments/3D-9_r3d18k_aniso512",
           "3D_v90z": "experiments/3D-5_r3d18k_vessel90z256", "3D_iso": "experiments/3D-1_r3d18k_iso256_mm"}


def main():
    seg = pd.read_csv(SEG / "evaluation_c90z_384/predictions.csv")
    labels = dict(zip(seg.exam_id, seg.true_label))
    test_ids, folds = label_only_folds(labels)
    dev = set().union(*folds)
    oof = {k: load_member(Path(v), folds)[0] for k, v in MEMBERS.items()}
    oof["nnunet"] = load_nnunet(SEG / "evaluation_c90z_384", dev, test_ids)[0]
    X = pd.DataFrame(oof).loc[sorted(dev)]
    y = np.array([labels[e] for e in X.index])
    fold_of = pd.Series({e: k for k, f in enumerate(folds) for e in f}).loc[X.index].to_numpy()

    def nested(rule, cols):
        p = np.empty(len(y))
        for k in range(len(folds)):
            tr, ev = fold_of != k, fold_of == k
            p[ev] = fit_rule(rule, X.loc[tr, cols], y[tr])(X.loc[ev, cols])
        return p

    print("single members:", {c: round(roc_auc_score(y, X[c]), 3) for c in X})
    print("Spearman correlation:\n", X.corr("spearman").round(2))
    rows = []
    for r in range(2, len(X.columns) + 1):
        for cols in map(list, combinations(X.columns, r)):
            for rule in ("mean", "stack"):
                p = nested(rule, cols)
                rows.append(("+".join(cols), rule, roc_auc_score(y, p), *paired_boot_diff(y, p, X["2D"].to_numpy())))
    df = pd.DataFrame(rows, columns=["members", "rule", "oof_auc", "minus_2D", "lo", "hi"])
    print(df.sort_values("oof_auc", ascending=False).round(3).to_string(index=False))

    age = pd.read_csv(DATASET_CSV).set_index("exam").Age.reindex(X.index)
    X["age"] = age.fillna(age.median()).rank(pct=True)  # 2 dev exams lack age
    for cols in (["2D", "3D_iso", "nnunet"], ["2D", "nnunet"], ["2D"]):
        for rule in ("mean", "stack"):
            with_age = nested(rule, cols + ["age"])
            without = X["2D"].to_numpy() if cols == ["2D"] else nested(rule, cols)
            print(f"age added to {'+'.join(cols)} ({rule}): {np.round(paired_boot_diff(y, with_age, without), 3)}")

    # Sensitivity of the frozen member list, reporting only: image-only 3D twin, and gender as an input
    X["3D_iso_image"] = load_member(Path("experiments/3D-2_r3d18k_iso256"), folds)[0].loc[X.index]
    gender = pd.read_csv(DATASET_CSV).set_index("exam").gender.reindex(X.index)
    print(f"dev exams lacking gender: {gender.isna().sum()}")
    X["gender"] = gender.fillna(gender.median()).rank(pct=True)
    base = ["2D", "3D_iso", "nnunet"]
    for name, cols in (("3D-1 swapped for image-only 3D-2", ["2D", "3D_iso_image", "nnunet"]), ("gender added", base + ["gender"])):
        for rule in ("mean", "stack"):
            p, b = nested(rule, cols), nested(rule, base)
            print(f"{name} ({rule}): {roc_auc_score(y, p):.3f} vs frozen {roc_auc_score(y, b):.3f}, diff {np.round(paired_boot_diff(y, p, b), 3)}")

    vox = {d: pd.read_csv(SEG / d / "predictions.csv").set_index("exam_id").n_positive_voxels.loc[X.index].to_numpy()
           for d in ("evaluation_c90z_384", "evaluation")}
    print("nnU-Net c90z_384 minus original NIfTI, dev only:",
          np.round(paired_boot_diff(y, vox["evaluation_c90z_384"], vox["evaluation"]), 3))


if __name__ == "__main__":
    main()
