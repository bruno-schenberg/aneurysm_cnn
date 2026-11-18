"""Score-level ensemble of final runs: choose on OOF, score the test set once.

Members are k-fold experiments run with EVAL_ON_TEST (val predictions.csv +
test_predictions.csv per fold, all on the label-only folds of
analysis/splits.py) and optionally the nnU-Net voxel count.

Scores are made comparable without labels: each fold's val scores are turned
into percentiles within that fold (probabilities aren't calibrated across
folds), and each test score into its percentile among the same fold model's
val scores, then averaged over folds. nnU-Net uses the dev distribution.

Rules, each estimated honestly by nested CV over the 5 folds (choose on 4,
evaluate on the 5th):
  mean     mean of all members
  greedy   forward selection by OOF AUC (Caruana-style, without replacement)
  stack    logistic regression on member percentiles
The rule with the best nested OOF AUC is refit on all dev folds and applied
once to the 114 test exams.

Usage (from training_engine/):
    python -m analysis.ensemble --member mip=experiments/X --member r3d=experiments/Y \\
        [--nnunet ../segmentation/evaluation_c90z_384] --out ensemble_out
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.bootstrap_auc import boot_ci, paired_boot_diff
from analysis.predict import exam_id
from analysis.splits import label_only_folds


def _percentile_of(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fraction of ref strictly below x, plus half of the ties (mid-rank)."""
    ref = np.sort(ref)
    return (np.searchsorted(ref, x, "left") + np.searchsorted(ref, x, "right")) / (2 * len(ref))


def load_member(exp_dir: Path, folds: list[set]) -> tuple[pd.Series, pd.Series]:
    """(dev OOF percentile per exam, test percentile per exam) for one k-fold experiment."""
    oof, test = {}, []
    for k, val_ids in enumerate(folds, start=1):
        v = pd.read_csv(exp_dir / f"fold_{k}/predictions.csv")
        v["exam_id"] = v.image.map(exam_id)
        if set(v.exam_id) != val_ids:
            raise ValueError(f"{exp_dir.name} fold_{k}: val exams differ from analysis/splits.py "
                             f"(was it run with STRATIFY_BY_SCANNER=false and SPLIT_SEED 42?)")
        ref = v.prob.to_numpy()
        oof.update(zip(v.exam_id, _percentile_of(ref, ref)))
        t = pd.read_csv(exp_dir / f"fold_{k}/test_predictions.csv")
        test.append(pd.Series(_percentile_of(ref, t.prob.to_numpy()), index=t.image.map(exam_id)))
    return pd.Series(oof), pd.concat(test, axis=1).mean(axis=1)


def load_nnunet(seg_dir: Path, dev_ids: set, test_ids: set) -> tuple[pd.Series, pd.Series]:
    s = pd.read_csv(seg_dir / "predictions.csv").set_index("exam_id").n_positive_voxels
    dev, test = s.loc[sorted(dev_ids)], s.loc[sorted(test_ids)]
    ref = dev.to_numpy()
    return (pd.Series(_percentile_of(ref, ref), index=dev.index),
            pd.Series(_percentile_of(ref, test.to_numpy()), index=test.index))


def _greedy(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    chosen, best = [], -1.0
    while True:
        cands = [(roc_auc_score(y, X[chosen + [c]].mean(axis=1)), c) for c in X if c not in chosen]
        if not cands:
            return chosen
        auc, c = max(cands)
        if auc <= best:
            return chosen
        chosen.append(c)
        best = auc


def fit_rule(rule: str, X: pd.DataFrame, y: np.ndarray):
    """Returns a function DataFrame -> scores, fitted on (X, y)."""
    if rule == "mean":
        return lambda Z: Z.mean(axis=1).to_numpy()
    if rule == "greedy":
        cols = _greedy(X, y)
        return lambda Z: Z[cols].mean(axis=1).to_numpy()
    if rule == "stack":
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
        return lambda Z: lr.predict_proba(Z[X.columns])[:, 1]
    raise ValueError(rule)


RULES = ("mean", "greedy", "stack")


def run(members: dict[str, Path], nnunet: Path | None, labels: dict[str, int]) -> dict:
    test_ids, folds = label_only_folds(labels)
    dev_ids = set().union(*folds)
    oof, test = {}, {}
    for name, d in members.items():
        oof[name], test[name] = load_member(d, folds)
    if nnunet is not None:
        oof["nnunet"], test["nnunet"] = load_nnunet(nnunet, dev_ids, test_ids)
    X = pd.DataFrame(oof).loc[sorted(dev_ids)]
    T = pd.DataFrame(test).loc[sorted(test_ids)]
    y = np.array([labels[e] for e in X.index])
    y_t = np.array([labels[e] for e in T.index])
    fold_of = pd.Series({e: k for k, f in enumerate(folds) for e in f}).loc[X.index].to_numpy()

    res = {"members_oof": {}, "rules_nested_oof": {}, "n_dev": len(y), "n_test": len(y_t),
           "n_test_pos": int(y_t.sum())}
    for c in X:
        res["members_oof"][c] = {"auc": roc_auc_score(y, X[c]), "ci": list(boot_ci(y, X[c].to_numpy()))}

    nested = {}
    for rule in RULES:
        p = np.empty(len(y))
        for k in range(len(folds)):
            tr, ev = fold_of != k, fold_of == k
            p[ev] = fit_rule(rule, X[tr], y[tr])(X[ev])
        nested[rule] = p
        res["rules_nested_oof"][rule] = {"auc": roc_auc_score(y, p), "ci": list(boot_ci(y, p))}
    best_rule = max(RULES, key=lambda r: res["rules_nested_oof"][r]["auc"])
    best_member = max(X, key=lambda c: res["members_oof"][c]["auc"])
    res["chosen_rule"], res["best_member_oof"] = best_rule, best_member
    res["chosen_vs_best_member_oof"] = list(paired_boot_diff(y, nested[best_rule], X[best_member].to_numpy()))
    if best_rule == "greedy":
        res["greedy_members"] = _greedy(X, y)

    # One shot on test: refit the chosen rule on all dev exams.
    p_test = fit_rule(best_rule, X, y)(T)
    res["test"] = {"ensemble": {"auc": roc_auc_score(y_t, p_test), "ci": list(boot_ci(y_t, p_test))},
                   "members": {c: {"auc": roc_auc_score(y_t, T[c])} for c in T},
                   "ensemble_minus_best_member": list(paired_boot_diff(y_t, p_test, T[best_member].to_numpy()))}
    if "nnunet" in T and best_member != "nnunet":
        res["test"]["ensemble_minus_nnunet"] = list(paired_boot_diff(y_t, p_test, T["nnunet"].to_numpy()))
    res["_frames"] = (X.assign(true_label=y, fold=fold_of, ensemble_nested=nested[best_rule]),
                      T.assign(true_label=y_t, ensemble=p_test))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--member", action="append", default=[], help="name=experiment_dir")
    p.add_argument("--nnunet", type=Path, help="nnU-Net evaluation dir with predictions.csv")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    members = dict(m.split("=", 1) for m in a.member)
    members = {k: Path(v) for k, v in members.items()}
    seg = pd.read_csv(Path(__file__).resolve().parents[2] / "segmentation/evaluation_c90z_384/predictions.csv")
    labels = dict(zip(seg.exam_id, seg.true_label))
    res = run(members, a.nnunet, labels)
    oof, test = res.pop("_frames")
    a.out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(a.out / "oof_percentiles.csv")
    test.to_csv(a.out / "test_percentiles.csv")
    (a.out / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("chosen_rule", "rules_nested_oof", "test")}, indent=2, default=float))


if __name__ == "__main__":
    main()
