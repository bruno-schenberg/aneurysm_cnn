"""Bootstrap 95% CIs for experiment AUCs.

Usage: python analysis/bootstrap_auc.py EXP [EXP ...]   (names under experiments/)
k-fold runs report both mean-of-folds AUC and pooled out-of-fold AUC.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            s.append(roc_auc_score(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])


def paired_boot_diff(y, p_a, p_b, n=2000, seed=0):
    """AUC(a) - AUC(b) on the same exams, with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            s.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return roc_auc_score(y, p_a) - roc_auc_score(y, p_b), *np.percentile(s, [2.5, 97.5])


def main(names):
    for name in names:
        folds = [pd.read_csv(f) for f in sorted(glob.glob(str(EXP_DIR / name / "fold_*/predictions.csv")))]
        per = [roc_auc_score(f.true_label, f.prob) for f in folds]
        d = pd.concat(folds)
        y, p = d.true_label.to_numpy(), d.prob.to_numpy()
        lo, hi = boot_ci(y, p)
        print(f"{name}: folds={len(folds)} mean={np.mean(per):.3f}±{np.std(per):.3f} "
              f"pooled={roc_auc_score(y, p):.3f} [{lo:.3f}, {hi:.3f}] n={len(y)} pos={y.sum()}")


if __name__ == "__main__":
    main(sys.argv[1:])
