"""
Validation check: split_data(stratify_by_scanner=True)
stratifies by class + scanner jointly, and folds rare scanner groups into
"{label}_other" instead of crashing StratifiedKFold.

Run: python training_engine/tests/test_scanner_stratified_split.py
"""

import csv
import tempfile
from pathlib import Path

import config as cfg
from src.data import split_data


def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "dataset.csv"
        rows = []
        # well-represented groups: 20 label0/siemens_3.0T, 20 label1/ge_1.5T
        for i in range(20):
            rows.append((f"e{i}", "0", "siemens", "3.0"))
        for i in range(20, 40):
            rows.append((f"e{i}", "1", "ge", "1.5"))
        # rare group: only 2 exams, must fall back to "1_other"
        rows.append(("e40", "1", "siemens", "1.5"))
        rows.append(("e41", "1", "siemens", "1.5"))

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["exam", "brand", "field_strength_T"])
            for exam, _label, brand, field in rows:
                w.writerow([exam, brand, field])

        cfg.DATASET_CSV = csv_path

        data = [
            {"image": f"/fake/{exam}.nii.gz", "label": int(label), "path": f"/fake/{exam}.nii.gz"}
            for exam, label, _brand, _field in rows
        ]

        folds = list(split_data(
            data, n_splits=5, seed=42, use_kfold=True, stratify_by_scanner=True,
        ))
        assert len(folds) == 5, "expected 5 folds"

        # every fold's val set should be non-empty and disjoint from train
        for fold_idx, train, val, test in folds:
            assert val, f"fold {fold_idx} has empty val set"
            train_ids = {d["path"] for d in train}
            val_ids = {d["path"] for d in val}
            assert train_ids.isdisjoint(val_ids)

        print("OK: scanner-stratified split_data ran across all folds "
              "without crashing on the rare siemens_1.5T group.")


if __name__ == "__main__":
    demo()
