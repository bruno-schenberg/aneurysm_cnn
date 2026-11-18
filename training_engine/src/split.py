"""Final-campaign split. Torch-free, so analysis/ can use it.

Used when a config sets EXCLUSIONS_FILE (dataset/exclusions.json):
- the exams in its "exclude" list are dropped (AneurysmExperiment.get_data_list)
- test = the frozen TEST_IDS_FILE exams; without one (Lausanne-only runs), one
  patient-grouped, label-stratified chunk of ~TEST_SPLIT_RATIO
- dev folds = StratifiedGroupKFold: all exams of a patient ("patient_groups",
  or the Lausanne sub-XXX) land in the same fold
- raises if a patient has exams in both test and dev
"""
import json
import os
import re
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
_BP = re.compile(r"BP\d{4}[A-Z]?")
_LAUS_SUB = re.compile(r"^laus_(sub-\d+)_")


def exam_key(path: str) -> str:
    """'BP0250A' for .../BP0250A.npy (any prefix), 'laus_sub-1_ses-2' for Lausanne."""
    name = os.path.basename(path)
    m = _BP.search(name)
    return m.group() if m else name.split(".")[0]


def load_exclusions(file: str | os.PathLike) -> tuple[set[str], dict[str, str]]:
    """(excluded exam keys, {exam: patient} for the listed multi-exam patients)."""
    d = json.loads((REPO / file).read_text())
    return set(d["exclude"]), {e: g[0] for g in d.get("patient_groups", []) for e in g}


def patient_of(key: str, patients: dict[str, str]) -> str:
    if key in patients:
        return patients[key]
    m = _LAUS_SUB.match(key)
    return f"laus_{m.group(1)}" if m else key


def grouped_split(keys, labels, groups, test_ids=None, n_splits=5, test_size=0.15,
                  seed=42, use_kfold=True, strata=None):
    """Yield (fold_idx, train_idx, val_idx, test_idx), indices into keys.

    Independent of input order (indices are walked in key order). With
    use_kfold=False only fold 0 is yielded (val = 1/n_splits of dev).
    strata (optional, e.g. scanner per exam) stratifies the dev folds on
    label x stratum; strata with fewer than n_splits dev exams become "other".
    The test set never depends on strata.
    """
    order = sorted(range(len(keys)), key=keys.__getitem__)
    if test_ids is not None:
        test = [i for i in order if keys[i] in test_ids]
        missing = set(test_ids) - {keys[i] for i in test}
        if missing:
            raise ValueError(f"{len(missing)} test exams missing from the data, e.g. {sorted(missing)[:5]}")
    else:
        sgkf = StratifiedGroupKFold(round(1 / test_size), shuffle=True, random_state=seed)
        _, t = next(sgkf.split(order, [labels[i] for i in order], [groups[i] for i in order]))
        test = [order[j] for j in t]
    in_test = set(test)
    dev = [i for i in order if i not in in_test]
    leak = {groups[i] for i in test} & {groups[i] for i in dev}
    if leak:
        raise ValueError(f"patients with exams in both test and dev: {sorted(leak)}")
    y = [labels[i] for i in dev]
    if strata is not None:
        y = [f"{labels[i]}_{strata[i]}" for i in dev]
        n = {k: y.count(k) for k in set(y)}
        y = [k if n[k] >= n_splits else f"{labels[i]}_other" for k, i in zip(y, dev)]
    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    folds = sgkf.split(dev, y, [groups[i] for i in dev])
    for k, (tr, va) in enumerate(folds):
        yield k, [dev[j] for j in tr], [dev[j] for j in va], test
        if not use_kfold:
            return
