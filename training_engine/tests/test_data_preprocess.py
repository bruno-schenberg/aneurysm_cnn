"""
Tests for data_preprocess.py — splitting, class weighting, and transforms.

All tests run without GPU, filesystem access to real NIfTI data, or a mounted
dataset directory. The NIfTI transform test writes a single synthetic file to
pytest's tmp_path fixture.
"""

import pytest
import numpy as np
import nibabel as nib
import torch

from src.data_preprocess import get_class_weights, get_transforms, split_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_data(n_negative: int, n_positive: int) -> list:
    """
    Builds a synthetic list of DataRecord dicts without filesystem access.
    Paths are unique strings sufficient for identity checks.
    """
    data = []
    for i in range(n_negative):
        data.append({"image": f"/fake/0/neg_{i}.nii.gz", "label": 0, "path": f"neg_{i}.nii.gz"})
    for i in range(n_positive):
        data.append({"image": f"/fake/1/pos_{i}.nii.gz", "label": 1, "path": f"pos_{i}.nii.gz"})
    return data


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:

    def test_same_seed_produces_identical_splits(self):
        """split_data called twice with the same seed yields identical fold contents."""
        data = make_synthetic_data(n_negative=40, n_positive=10)

        splits_a = list(split_data(data, n_splits=5, test_size=0.2, seed=42))
        splits_b = list(split_data(data, n_splits=5, test_size=0.2, seed=42))

        assert len(splits_a) == len(splits_b) == 5
        for (fold_a, train_a, val_a, test_a), (fold_b, train_b, val_b, test_b) in zip(
            splits_a, splits_b
        ):
            assert fold_a == fold_b
            assert [d["path"] for d in train_a] == [d["path"] for d in train_b]
            assert [d["path"] for d in val_a] == [d["path"] for d in val_b]
            assert [d["path"] for d in test_a] == [d["path"] for d in test_b]

    def test_different_seeds_produce_different_splits(self):
        """Different seeds must produce at least one different fold split."""
        data = make_synthetic_data(n_negative=40, n_positive=10)

        splits_42 = list(split_data(data, n_splits=5, test_size=0.2, seed=42))
        splits_99 = list(split_data(data, n_splits=5, test_size=0.2, seed=99))

        any_difference = any(
            [d["path"] for d in train_42] != [d["path"] for d in train_99]
            for (_, train_42, _, _), (_, train_99, _, _) in zip(splits_42, splits_99)
        )
        assert any_difference, "Different seeds produced identical splits — seeding may be broken."


# ---------------------------------------------------------------------------
# Data Integrity (no leakage)
# ---------------------------------------------------------------------------


class TestDataIntegrity:

    def test_train_and_val_are_disjoint_within_every_fold(self):
        """No case appears in both train and val sets within any fold."""
        data = make_synthetic_data(n_negative=40, n_positive=10)
        for _, train_files, val_files, _ in split_data(
            data, n_splits=5, test_size=0.2, seed=42
        ):
            train_paths = {d["path"] for d in train_files}
            val_paths = {d["path"] for d in val_files}
            assert train_paths.isdisjoint(val_paths), (
                f"Leakage detected: {train_paths & val_paths}"
            )

    def test_test_set_is_disjoint_from_all_train_and_val_sets(self):
        """The hold-out test set never overlaps with any fold's train or val set."""
        data = make_synthetic_data(n_negative=40, n_positive=10)
        for _, train_files, val_files, test_files in split_data(
            data, n_splits=5, test_size=0.2, seed=42
        ):
            test_paths = {d["path"] for d in test_files}
            assert test_paths.isdisjoint({d["path"] for d in train_files}), (
                "Test set overlaps with training set."
            )
            assert test_paths.isdisjoint({d["path"] for d in val_files}), (
                "Test set overlaps with validation set."
            )

    def test_test_set_is_identical_across_all_folds(self):
        """The hold-out test set is the same for every fold (created once)."""
        data = make_synthetic_data(n_negative=40, n_positive=10)
        test_sets = [
            frozenset(d["path"] for d in test_files)
            for _, _, _, test_files in split_data(data, n_splits=5, test_size=0.2, seed=42)
        ]
        assert len(set(test_sets)) == 1, "Test set content varies across folds."

    def test_correct_number_of_folds_generated(self):
        """split_data yields exactly n_splits folds."""
        data = make_synthetic_data(n_negative=40, n_positive=10)
        splits = list(split_data(data, n_splits=5, test_size=0.2, seed=42))
        assert len(splits) == 5


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


class TestStratification:

    def test_val_fold_preserves_approximate_class_ratio(self):
        """Each fold's val set has approximately the same positive rate as the full dataset."""
        data = make_synthetic_data(n_negative=80, n_positive=20)
        expected_positive_rate = 20 / 100

        for _, _, val_files, _ in split_data(data, n_splits=5, test_size=0.2, seed=42):
            val_positives = sum(1 for d in val_files if d["label"] == 1)
            val_positive_rate = val_positives / len(val_files)
            assert abs(val_positive_rate - expected_positive_rate) < 0.10, (
                f"Val positive rate {val_positive_rate:.2f} deviates too far from "
                f"expected {expected_positive_rate:.2f}."
            )


# ---------------------------------------------------------------------------
# Class Weighting
# ---------------------------------------------------------------------------


class TestClassWeighting:

    def test_weights_inversely_proportional_to_class_frequency(self):
        """Minority class receives a proportionally higher weight."""
        # 3 positives, 9 negatives → weight_1 / weight_0 should equal 9 / 3 = 3.0
        data = make_synthetic_data(n_negative=9, n_positive=3)
        weights = get_class_weights(data)

        assert weights is not None
        assert weights.shape == torch.Size([2])
        assert weights[1] > weights[0], "Minority class should have higher weight."

        expected_ratio = 9.0 / 3.0
        actual_ratio = weights[1].item() / weights[0].item()
        assert abs(actual_ratio - expected_ratio) < 1e-5

    def test_balanced_data_produces_equal_weights(self):
        """Perfectly balanced data produces equal weights for both classes."""
        data = make_synthetic_data(n_negative=10, n_positive=10)
        weights = get_class_weights(data)

        assert weights is not None
        assert abs(weights[0].item() - weights[1].item()) < 1e-5

    def test_empty_data_returns_none(self):
        """get_class_weights returns None for an empty list."""
        assert get_class_weights([]) is None

    def test_weights_are_float_tensor(self):
        """Returned weights tensor has dtype float."""
        data = make_synthetic_data(n_negative=6, n_positive=4)
        weights = get_class_weights(data)

        assert weights is not None
        assert weights.dtype == torch.float32


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:

    def test_output_shape_is_1_128_128_128(self, tmp_path):
        """Eval transform resizes any NIfTI to (1, 128, 128, 128)."""
        synthetic_volume = np.random.rand(32, 32, 32).astype(np.float32)
        nifti_img = nib.Nifti1Image(synthetic_volume, affine=np.eye(4))
        nifti_path = str(tmp_path / "synthetic.nii.gz")
        nib.save(nifti_img, nifti_path)

        transform = get_transforms(spatial_size=(128, 128, 128), augment=False)
        result = transform({"image": nifti_path})

        assert result["image"].shape == torch.Size([1, 128, 128, 128])

    def test_output_values_in_zero_one_range(self, tmp_path):
        """ScaleIntensity normalises output values to [0, 1]."""
        synthetic_volume = (np.random.rand(32, 32, 32) * 1000).astype(np.float32)
        nifti_img = nib.Nifti1Image(synthetic_volume, affine=np.eye(4))
        nifti_path = str(tmp_path / "synthetic_high_intensity.nii.gz")
        nib.save(nifti_img, nifti_path)

        transform = get_transforms(spatial_size=(128, 128, 128), augment=False)
        result = transform({"image": nifti_path})
        tensor = result["image"]

        assert tensor.min().item() >= 0.0 - 1e-6
        assert tensor.max().item() <= 1.0 + 1e-6

    def test_tabular_key_becomes_tensor_when_use_tabular(self, tmp_path):
        """With use_tabular=True, the 'tabular' key in the sample is converted to a tensor."""
        synthetic_volume = np.random.rand(32, 32, 32).astype(np.float32)
        nifti_img = nib.Nifti1Image(synthetic_volume, affine=np.eye(4))
        nifti_path = str(tmp_path / "synthetic.nii.gz")
        nib.save(nifti_img, nifti_path)

        transform = get_transforms(spatial_size=(128, 128, 128), augment=False, use_tabular=True)
        tabular_array = np.array([0.45, 1.0], dtype=np.float32)  # age=45/100, gender=1
        result = transform({"image": nifti_path, "tabular": tabular_array})

        assert "tabular" in result
        assert isinstance(result["tabular"], torch.Tensor)
        assert result["tabular"].shape == torch.Size([2])

    def test_tabular_values_preserved_after_transform(self, tmp_path):
        """Transform must not alter the numeric values in the 'tabular' field."""
        synthetic_volume = np.random.rand(32, 32, 32).astype(np.float32)
        nifti_img = nib.Nifti1Image(synthetic_volume, affine=np.eye(4))
        nifti_path = str(tmp_path / "synthetic.nii.gz")
        nib.save(nifti_img, nifti_path)

        transform = get_transforms(spatial_size=(128, 128, 128), augment=False, use_tabular=True)
        tabular_array = np.array([0.62, 0.0], dtype=np.float32)
        result = transform({"image": nifti_path, "tabular": tabular_array})

        assert abs(result["tabular"][0].item() - 0.62) < 1e-5
        assert abs(result["tabular"][1].item() - 0.0) < 1e-5


# ---------------------------------------------------------------------------
# Tabular data loading
# ---------------------------------------------------------------------------


class TestTabularDataLoading:

    def _write_nifti(self, path):
        vol = np.random.rand(8, 8, 8).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), path)

    def _make_dataset_dir(self, tmp_path, case_ids_by_label):
        """Creates class subdirs with dummy NIfTI files. Returns root dir."""
        root = tmp_path / "dataset"
        for label, case_ids in case_ids_by_label.items():
            class_dir = root / str(label)
            class_dir.mkdir(parents=True)
            for cid in case_ids:
                self._write_nifti(str(class_dir / f"{cid}.nii.gz"))
        return str(root)

    def _write_csv(self, tmp_path, rows):
        """Writes a tabular CSV and returns its path."""
        csv_path = str(tmp_path / "metadata.csv")
        with open(csv_path, "w") as f:
            f.write("case_id,age,gender\n")
            for case_id, age, gender in rows:
                f.write(f"{case_id},{age},{gender}\n")
        return csv_path

    def test_get_data_list_without_csv_has_no_tabular_key(self, tmp_path):
        """Without a CSV, records must not contain a 'tabular' key."""
        from src.data_preprocess import get_data_list
        root = self._make_dataset_dir(tmp_path, {0: ["case_A"], 1: ["case_B"]})
        data = get_data_list(root)
        for record in data:
            assert "tabular" not in record

    def test_get_data_list_with_csv_adds_tabular_key(self, tmp_path):
        """With a CSV, every record must have a 'tabular' key."""
        from src.data_preprocess import get_data_list
        root = self._make_dataset_dir(tmp_path, {0: ["case_A"], 1: ["case_B"]})
        csv_path = self._write_csv(tmp_path, [("case_A", 45, 0), ("case_B", 62, 1)])
        data = get_data_list(root, tabular_csv=csv_path)
        for record in data:
            assert "tabular" in record

    def test_tabular_array_shape_is_2(self, tmp_path):
        """Each 'tabular' array must have exactly 2 elements: [age/100, gender]."""
        from src.data_preprocess import get_data_list
        root = self._make_dataset_dir(tmp_path, {0: ["case_A"], 1: ["case_B"]})
        csv_path = self._write_csv(tmp_path, [("case_A", 45, 0), ("case_B", 62, 1)])
        data = get_data_list(root, tabular_csv=csv_path)
        for record in data:
            assert record["tabular"].shape == (2,)

    def test_age_is_normalised_by_100(self, tmp_path):
        """Age in the tabular array must be age/100, not raw years."""
        from src.data_preprocess import get_data_list
        root = self._make_dataset_dir(tmp_path, {0: ["case_A"]})
        csv_path = self._write_csv(tmp_path, [("case_A", 50, 0)])
        data = get_data_list(root, tabular_csv=csv_path)
        age_normalised = data[0]["tabular"][0]
        assert abs(age_normalised - 0.5) < 1e-5

    def test_gender_stored_as_float(self, tmp_path):
        """Gender must be stored as a float (0.0 or 1.0)."""
        from src.data_preprocess import get_data_list
        root = self._make_dataset_dir(tmp_path, {0: ["case_A"], 1: ["case_B"]})
        csv_path = self._write_csv(tmp_path, [("case_A", 30, 0), ("case_B", 40, 1)])
        data = get_data_list(root, tabular_csv=csv_path)
        for record in data:
            assert record["tabular"].dtype == np.float32

    def test_missing_case_in_csv_raises_key_error(self, tmp_path):
        """A NIfTI file with no matching CSV row must raise KeyError with the case id."""
        from src.data_preprocess import get_data_list
        root = self._make_dataset_dir(tmp_path, {0: ["case_A", "case_MISSING"]})
        csv_path = self._write_csv(tmp_path, [("case_A", 30, 1)])
        with pytest.raises(KeyError, match="case_MISSING"):
            get_data_list(root, tabular_csv=csv_path)
