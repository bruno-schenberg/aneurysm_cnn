"""Exam-level classifier on nnU-Net c90z_384 outputs.

The voxel count throws away what nnU-Net knows about *where* and *how many*
candidates there are. This fits a small model on per-exam candidate features,
using the same label-only 5-fold split as the final training runs (OOF on the
645 dev exams), then scores the 114 test exams once with a model fit on all dev.

Usage (from repo root, light analysis env):
    training_engine/.venv/bin/python segmentation/feature_classifier.py
Outputs: segmentation/evaluation_c90z_384/feature_classifier_{predictions.csv,summary.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_engine"))
from analysis.bootstrap_auc import boot_ci, paired_boot_diff  # noqa: E402
from analysis.splits import label_only_folds  # noqa: E402

EVAL = ROOT / "segmentation/evaluation_c90z_384"


def exam_features() -> pd.DataFrame:
    pred = pd.read_csv(EVAL / "predictions.csv").set_index("exam_id")
    cand = pd.read_csv(EVAL / "candidates.csv").sort_values("voxel_count", ascending=False)
    g = cand.groupby("exam_id")
    largest = g.first()
    f = pd.DataFrame(index=pred.index)
    f["label"] = pred.true_label
    f["log_total_voxels"] = np.log1p(pred.n_positive_voxels)
    f["n_components"] = g.size().reindex(f.index, fill_value=0)
    f["log_largest"] = np.log1p(largest.voxel_count.reindex(f.index, fill_value=0))
    f["log_second"] = np.log1p(g.voxel_count.nth(1).groupby(level=0).first()
                               .reindex(f.index, fill_value=0)) if g.size().max() > 1 else 0.0
    # Position of the largest candidate (c90z_384 voxel coords); exams without one
    # get the dataset mean so has_candidate carries that information instead.
    for c in ("cz", "cy", "cx"):
        f[f"largest_{c}"] = largest[c].reindex(f.index).fillna(largest[c].mean())
    f["has_candidate"] = (f.n_components > 0).astype(int)
    return f


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)),
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=10, random_state=0),
}


def main():
    f = exam_features()
    X_cols = [c for c in f.columns if c != "label"]
    test_ids, folds = label_only_folds(f.label.to_dict())
    dev = f.drop(index=list(test_ids))
    test = f.loc[sorted(test_ids)]

    out = pd.DataFrame({"exam_id": f.index, "true_label": f.label.values,
                        "voxel_count_score": f.log_total_voxels.values}).set_index("exam_id")
    out["split"] = np.where(out.index.isin(test_ids), "test", "dev")
    out["fold"] = -1
    for k, val in enumerate(folds):
        out.loc[sorted(val), "fold"] = k

    summary = {}
    y_dev = dev.label.to_numpy()
    base = dev.log_total_voxels.to_numpy()
    lo, hi = boot_ci(y_dev, base)
    summary["voxel_count"] = {"oof_auc": roc_auc_score(y_dev, base), "ci": [lo, hi]}
    print(f"voxel count (dev, n={len(dev)}): AUC {summary['voxel_count']['oof_auc']:.3f} [{lo:.3f}, {hi:.3f}]")

    for name, make in MODELS.items():
        oof = pd.Series(index=dev.index, dtype=float)
        for val in folds:
            tr = dev.drop(index=list(val))
            m = make().fit(tr[X_cols], tr.label)
            oof.loc[sorted(val)] = m.predict_proba(dev.loc[sorted(val), X_cols])[:, 1]
        p = oof.loc[dev.index].to_numpy()
        auc = roc_auc_score(y_dev, p)
        lo, hi = boot_ci(y_dev, p)
        d, dlo, dhi = paired_boot_diff(y_dev, p, base)
        summary[name] = {"oof_auc": auc, "ci": [lo, hi], "minus_voxel_count": [d, dlo, dhi]}
        print(f"{name:7s} (dev OOF): AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  vs voxel count {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        out.loc[dev.index, f"{name}_prob"] = p
        out.loc[test.index, f"{name}_prob"] = make().fit(dev[X_cols], dev.label).predict_proba(test[X_cols])[:, 1]

    # Test set: scored and saved, but the model choice above is made on OOF only.
    y_t = test.label.to_numpy()
    for col in ["voxel_count_score"] + [f"{n}_prob" for n in MODELS]:
        p = out.loc[test.index, col].to_numpy()
        lo, hi = boot_ci(y_t, p)
        summary.setdefault("test", {})[col] = {"auc": roc_auc_score(y_t, p), "ci": [lo, hi], "n": len(y_t), "pos": int(y_t.sum())}

    out.reset_index().to_csv(EVAL / "feature_classifier_predictions.csv", index=False)
    (EVAL / "feature_classifier_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved {EVAL / 'feature_classifier_predictions.csv'} and summary.json (test AUCs inside, not printed)")


if __name__ == "__main__":
    main()
