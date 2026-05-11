"""
Tests for data_preprocess.py — splitting, class weighting, transforms, and
real-data integration.

Unit tests (splitting, weighting, config) run entirely in-memory with synthetic
data — no GPU, no real NIfTI files, no mounted dataset required.

Integration tests (TestTransforms, TestRealDataPipeline) load real NIfTI files
from the 12-case sample dataset defined in conftest.py.  They are skipped
automatically when the dataset is not mounted.
"""

import pytest
import numpy as np
import nibabel as nib
import torch

from src.data_preprocess import build_dataloaders, get_class_weights, get_data_list, get_transforms, split_data, parse_input_resolution


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
# parse_input_resolution
# ---------------------------------------------------------------------------


class TestParseInputResolution:
    """parse_input_resolution must map valid strings to tuples and reject bad input."""

    def test_128x128x128_returns_correct_tuple(self):
        """'128x128x128' must return (128, 128, 128)."""
        assert parse_input_resolution("128x128x128") == (128, 128, 128)

    def test_256x256x176_returns_correct_tuple(self):
        """'256x256x176' must return (256, 256, 176)."""
        assert parse_input_resolution("256x256x176") == (256, 256, 176)

    def test_256x256x256_returns_correct_tuple(self):
        """'256x256x256' must return (256, 256, 256)."""
        assert parse_input_resolution("256x256x256") == (256, 256, 256)

    def test_unrecognised_string_raises_value_error(self):
        """An unrecognised resolution string must raise ValueError."""
        with pytest.raises(ValueError):
            parse_input_resolution("512x512x512")

    def test_error_message_is_informative(self):
        """The ValueError message must include the invalid string and valid options."""
        with pytest.raises(ValueError, match="bad_value"):
            parse_input_resolution("bad_value")


# ---------------------------------------------------------------------------
# build_dataloaders spatial_size propagation
# ---------------------------------------------------------------------------


class TestBuildDataloadersResolution:
    """build_dataloaders must accept spatial_size and produce batches of that shape."""

    def _make_nifti(self, path, shape=(16, 16, 16)):
        vol = np.random.rand(*shape).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), path)

    def _make_dataset(self, tmp_path, n_neg=4, n_pos=4):
        root = tmp_path / "ds"
        for label in [0, 1]:
            d = root / str(label)
            d.mkdir(parents=True)
            count = n_neg if label == 0 else n_pos
            for i in range(count):
                self._make_nifti(str(d / f"case_{i}.nii.gz"))
        return str(root)

    def test_spatial_size_64_produces_correct_batch_shape(self, tmp_path):
        """build_dataloaders with spatial_size=(64,64,64) must yield (B,1,64,64,64) batches."""
        from src.data_preprocess import get_data_list, split_data

        root = self._make_dataset(tmp_path)
        data = get_data_list(root)
        splits = list(split_data(data, n_splits=2, test_size=0.2, seed=42, use_kfold=False, val_split_ratio=0.25))
        _, train_files, val_files, test_files = splits[0]

        train_loader, _, _ = build_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=2,
            val_batch_size=2,
            seed=42,
            spatial_size=(64, 64, 64),
        )

        batch = next(iter(train_loader))
        assert batch["image"].shape[1:] == torch.Size([1, 64, 64, 64]), (
            f"Unexpected shape: {batch['image'].shape}"
        )

    def test_default_spatial_size_is_128(self, tmp_path):
        """build_dataloaders without spatial_size must default to (128,128,128)."""
        from src.data_preprocess import get_data_list, split_data

        root = self._make_dataset(tmp_path)
        data = get_data_list(root)
        splits = list(split_data(data, n_splits=2, test_size=0.2, seed=42, use_kfold=False, val_split_ratio=0.25))
        _, train_files, val_files, test_files = splits[0]

        train_loader, _, _ = build_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=2,
            val_batch_size=2,
            seed=42,
        )

        batch = next(iter(train_loader))
        assert batch["image"].shape[1:] == torch.Size([1, 128, 128, 128])


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
# Transforms  (use real NIfTI files from the sample dataset)
# ---------------------------------------------------------------------------


class TestTransforms:

    def test_output_shape_is_1_128_128_128(self, sample_nifti_neg):
        """Eval transform produces shape (1, 128, 128, 128) from a real NIfTI file."""
        transform = get_transforms(spatial_size=(128, 128, 128), augment=False)
        result = transform({"image": str(sample_nifti_neg)})
        assert result["image"].shape == torch.Size([1, 128, 128, 128])

    def test_output_values_in_zero_one_range(self, sample_nifti_neg):
        """ScaleIntensity normalises real voxel intensities to [0, 1]."""
        transform = get_transforms(spatial_size=(128, 128, 128), augment=False)
        result = transform({"image": str(sample_nifti_neg)})
        tensor = result["image"]
        assert tensor.min().item() >= 0.0 - 1e-6
        assert tensor.max().item() <= 1.0 + 1e-6

    def test_tabular_key_becomes_tensor_when_use_tabular(self, sample_nifti_neg):
        """With use_tabular=True, the 'tabular' key is converted to a tensor."""
        transform = get_transforms(spatial_size=(128, 128, 128), augment=False, use_tabular=True)
        tabular_array = np.array([0.45, 1.0], dtype=np.float32)  # age=45/100, gender=1
        result = transform({"image": str(sample_nifti_neg), "tabular": tabular_array})
        assert "tabular" in result
        assert isinstance(result["tabular"], torch.Tensor)
        assert result["tabular"].shape == torch.Size([2])

    def test_tabular_values_preserved_after_transform(self, sample_nifti_neg):
        """Transform must not alter the numeric values in the 'tabular' field."""
        transform = get_transforms(spatial_size=(128, 128, 128), augment=False, use_tabular=True)
        tabular_array = np.array([0.62, 0.0], dtype=np.float32)
        result = transform({"image": str(sample_nifti_neg), "tabular": tabular_array})
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
        """Writes a tabular CSV in classes.csv format and returns its path."""
        csv_path = str(tmp_path / "metadata.csv")
        with open(csv_path, "w") as f:
            f.write("exam,Age,gender\n")
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


# ---------------------------------------------------------------------------
# Real-data integration  (requires sample dataset — skipped if not mounted)
# ---------------------------------------------------------------------------


class TestRealDataPipeline:
    """
    End-to-end integration tests using real 128×128×128 NIfTI files from the
    12-case variant-D sample dataset.

    These tests exercise the full data-loading chain:
        get_data_list → split_data → build_dataloaders → DataLoader iteration

    They are skipped automatically when the dataset is not mounted (the
    ``sample_dataset_path`` fixture calls ``pytest.skip`` in that case).
    """

    def test_get_data_list_finds_all_12_cases(self, sample_dataset_path):
        """get_data_list discovers all 12 NIfTI files and assigns correct labels."""
        data = get_data_list(str(sample_dataset_path))
        assert len(data) == 12

    def test_get_data_list_label_counts(self, sample_dataset_path):
        """6 negatives and 6 positives are present in the sample dataset."""
        data = get_data_list(str(sample_dataset_path))
        negatives = sum(1 for d in data if d["label"] == 0)
        positives = sum(1 for d in data if d["label"] == 1)
        assert negatives == 6
        assert positives == 6

    def test_get_data_list_records_have_required_keys(self, sample_dataset_path):
        """Every record must contain 'image', 'label', and 'path' keys."""
        data = get_data_list(str(sample_dataset_path))
        for record in data:
            assert "image" in record
            assert "label" in record
            assert "path" in record

    def test_split_data_produces_disjoint_sets(self, sample_dataset_path):
        """Train, val, and test sets must be mutually disjoint on the real dataset."""
        data = get_data_list(str(sample_dataset_path))
        for _, train_files, val_files, test_files in split_data(
            data, n_splits=2, test_size=0.2, seed=42
        ):
            train_paths = {d["path"] for d in train_files}
            val_paths = {d["path"] for d in val_files}
            test_paths = {d["path"] for d in test_files}
            assert train_paths.isdisjoint(val_paths), "Train/val overlap on real dataset."
            assert train_paths.isdisjoint(test_paths), "Train/test overlap on real dataset."
            assert val_paths.isdisjoint(test_paths), "Val/test overlap on real dataset."

    def test_build_dataloaders_batch_image_shape(self, sample_dataset_path):
        """
        build_dataloaders must produce batches with image shape (B, 1, 128, 128, 128).

        Uses a single non-kfold split so the test runs quickly without iterating
        all folds.  batch_size=2 to verify the batch dimension is assembled correctly.
        """
        data = get_data_list(str(sample_dataset_path))
        splits = list(split_data(
            data, n_splits=2, test_size=0.2, seed=42, use_kfold=False, val_split_ratio=0.25
        ))
        _, train_files, val_files, test_files = splits[0]

        train_loader, val_loader, _ = build_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=2,
            val_batch_size=2,
            seed=42,
        )

        batch = next(iter(train_loader))
        assert batch["image"].shape[1:] == torch.Size([1, 128, 128, 128]), (
            f"Unexpected image shape: {batch['image'].shape}"
        )

    def test_build_dataloaders_batch_labels_are_integer(self, sample_dataset_path):
        """Labels in each batch must be integer tensors (0 or 1)."""
        data = get_data_list(str(sample_dataset_path))
        splits = list(split_data(
            data, n_splits=2, test_size=0.2, seed=42, use_kfold=False, val_split_ratio=0.25
        ))
        _, train_files, val_files, test_files = splits[0]

        train_loader, _, _ = build_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=2,
            val_batch_size=2,
            seed=42,
        )

        batch = next(iter(train_loader))
        labels = batch["label"]
        assert labels.dtype in (torch.int32, torch.int64, torch.long)
        assert all(l.item() in (0, 1) for l in labels)


# ---------------------------------------------------------------------------
# build_dataloaders — cache_rate parameter
# ---------------------------------------------------------------------------


class TestBuildDataloadersCacheRate:
    """
    build_dataloaders must use CacheDataset when cache_rate > 0 and fall back to
    the plain Dataset when cache_rate == 0.  These tests are structural (they
    check the dataset type without loading real files), so they run in-process
    with no GPU or mounted dataset required.
    """

    def _fake_files(self, n: int = 2) -> list:
        """Return minimal records; the dataset type check doesn't load files."""
        return [{"image": f"/fake/{i}.nii.gz", "label": i % 2, "path": f"{i}.nii.gz"} for i in range(n)]

    def test_cache_rate_zero_uses_plain_dataset(self):
        """cache_rate=0.0 must produce plain MONAI Dataset objects (lazy loading)."""
        from monai.data import Dataset
        from monai.data import CacheDataset
        train_loader, val_loader, test_loader = build_dataloaders(
            train_files=self._fake_files(4),
            val_files=self._fake_files(2),
            test_files=self._fake_files(2),
            batch_size=2,
            val_batch_size=2,
            seed=42,
            cache_rate=0.0,
        )
        assert isinstance(train_loader.dataset, Dataset)
        assert not isinstance(train_loader.dataset, CacheDataset)

    def test_cache_rate_positive_uses_cache_dataset(self, monkeypatch):
        """cache_rate=1.0 must wrap the data in a MONAI CacheDataset.

        CacheDataset fills its cache at __init__ time, which would try to load
        real NIfTI files.  We patch _fill_cache to a no-op so the structural
        check runs without filesystem access, then verify the dataset type.
        """
        from monai.data import CacheDataset
        monkeypatch.setattr(CacheDataset, "_fill_cache", lambda *a, **kw: None)

        train_loader, val_loader, test_loader = build_dataloaders(
            train_files=self._fake_files(4),
            val_files=self._fake_files(2),
            test_files=self._fake_files(2),
            batch_size=2,
            val_batch_size=2,
            seed=42,
            cache_rate=1.0,
        )
        assert isinstance(train_loader.dataset, CacheDataset)
        assert isinstance(val_loader.dataset, CacheDataset)
        assert isinstance(test_loader.dataset, CacheDataset)
