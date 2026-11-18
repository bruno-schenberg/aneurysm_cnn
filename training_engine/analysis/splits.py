"""The folds every final run uses, rebuilt without importing torch.

Same rule as AneurysmExperiment.split_data with EXCLUSIONS_FILE set
(src/split.py): excluded exams dropped, frozen dataset/test_ids.json test set,
patient-grouped StratifiedGroupKFold on the rest. analysis/ensemble.py checks
every member's val sets against these, so a mismatch fails loudly.
"""
import json
from pathlib import Path

from src.split import grouped_split, load_exclusions, patient_of

TEST_IDS = Path(__file__).resolve().parents[2] / "dataset/test_ids.json"
EXCLUSIONS = "dataset/exclusions.json"


def label_only_folds(labels: dict[str, int], n_splits=5, seed=42):
    """{exam_id: label} -> (test_ids, [val_ids of fold 0..n-1]). Excluded exams are dropped."""
    excluded, patients = load_exclusions(EXCLUSIONS)
    keys = sorted(e for e in labels if e not in excluded)
    test_ids = set(json.loads(TEST_IDS.read_text()))
    folds = grouped_split(keys, [labels[k] for k in keys], [patient_of(k, patients) for k in keys],
                          test_ids=test_ids, n_splits=n_splits, seed=seed)
    return test_ids, [{keys[i] for i in va} for _, _, va, _ in folds]
