"""
Tests for data_preprocess.py — splitting, class weighting, and transforms.

All tests run without GPU, filesystem access to real NIfTI data, or a mounted
dataset directory. The NIfTI transform test writes a single synthetic file to
pytest's tmp_path fixture.
"""

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
