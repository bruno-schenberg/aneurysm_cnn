"""D10.7: test-set comparison of every final 2D config against the final model (2D-3_512).

Seed repeats (_s1, _s2) are grouped with their base run. Per config: mean val AUC and
mean per-fold test AUC over seeds, plus the test AUC of the fold×seed ensemble (mean prob
per exam) with a bootstrap CI and a paired difference against the final model's ensemble.
Reporting only: configs were chosen on val, never on these numbers.

Usage (from training_engine/): python -m analysis.test_comparison_2d
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.bootstrap_auc import boot_ci, paired_boot_diff
from analysis.predict import exam_id

EXP = Path("experiments")
FINAL = "2D-3_512"


def test_probs(run_dirs):
    t = pd.concat(pd.read_csv(f) for d in run_dirs for f in sorted(d.glob("fold_*/test_predictions.csv")))
    t["exam_id"] = t.image.map(exam_id)
    return t.groupby("exam_id").agg(y=("true_label", "first"), p=("prob", "mean"))


def main():
    groups = {}
    for d in sorted(EXP.glob("2D-*")):
        if (d / "summary.json").exists():
            groups.setdefault(re.sub(r"_s\d+$", "", d.name), []).append(d)
    final = test_probs(groups[FINAL])
    rows = []
    for name, dirs in groups.items():
        s = [json.loads((d / "summary.json").read_text()) for d in dirs]
        t = test_probs(dirs)
        if not t.index.isin(final.index).all():
            print(f"skip {name}: tested on a different test set")  # 2D-6_lausanne_only uses a Lausanne hold-out
            continue
        t = t.loc[final.index]
        assert (t.y == final.y).all()
        y, p = t.y.to_numpy(), t.p.to_numpy()
        diff = paired_boot_diff(y, p, final.p.to_numpy()) if name != FINAL else (0.0, 0.0, 0.0)
        rows.append((name, len(dirs), np.mean([x["mean_auc"] for x in s]), np.mean([x["test"]["mean_auc"] for x in s]),
                     roc_auc_score(y, p), *boot_ci(y, p), *diff))
    df = pd.DataFrame(rows, columns=["config", "seeds", "val", "test_mean_folds", "test_ens", "lo", "hi",
                                     "minus_final", "d_lo", "d_hi"])
    pd.set_option("display.width", 200)
    print(f"n_test={len(final)}, positives={int(final.y.sum())}")
    print(df.sort_values("val", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
