"""
data_preprocess.py

Handles all data preparation for the training engine: NIfTI file discovery,
stratified data splitting, class-imbalance weighting, MONAI transform construction,
and DataLoader assembly.

- All volumes are 128×128×128 float32 NIfTIs in ``0/`` (negative) or ``1/`` (positive)
  subdirectories by the time this module runs.
- Weighted loss and oversampling are two complementary imbalance strategies; both
  aim to prevent the model from collapsing to the majority class.
- Three RNG streams are seeded (Python, NumPy, PyTorch) plus MONAI's ``set_determinism``.
"""

import csv
import os
import random
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    Resized,
    ScaleIntensityd,
)
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedKFold, train_test_split


# Single source of truth for all valid INPUT_RESOLUTION strings.
# Maps the human-readable "HxWxD" config key to the (H, W, D) tuple that
# MONAI's Resized transform expects. Adding a new resolution requires only
# a new entry here — no other module needs to be modified.
VALID_RESOLUTIONS: Dict[str, Tuple[int, int, int]] = {
    "128x128x128": (128, 128, 128),
    "192x192x128": (192, 192, 128),
    "256x256x176": (256, 256, 176),
    "256x256x256": (256, 256, 256),
}


def parse_input_resolution(resolution: str) -> Tuple[int, int, int]:
    """
    Parse an ``INPUT_RESOLUTION`` config string into a spatial-size tuple.

    Args:
        resolution: A resolution string that must be a key in
            ``VALID_RESOLUTIONS`` (e.g. ``"128x128x128"``).

    Returns:
        A ``(H, W, D)`` integer tuple suitable for passing to MONAI's
        ``Resized`` transform as ``spatial_size``.

    Raises:
        ValueError: If ``resolution`` is not a key in ``VALID_RESOLUTIONS``.
    """
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid INPUT_RESOLUTION '{resolution}'. "
            f"Valid options: {list(VALID_RESOLUTIONS.keys())}"
        )
    return VALID_RESOLUTIONS[resolution]


# ----------------------------------------------------
# 1. Global Reproducibility
# ----------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """
    Seed all random number generators and enable MONAI/cuDNN deterministic mode.

    Covers five independent RNG sources: Python's ``random``, NumPy, PyTorch CPU,
    PyTorch CUDA, and MONAI's ``set_determinism``. ``cudnn.deterministic = True``
    is set for reproducible kernel selection; ``benchmark`` is left enabled for
    auto-tuning performance on 3D convolutions.

    Args:
        seed: Integer seed applied to all RNG sources. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """
    Seed the random state of each DataLoader worker process.

    PyTorch gives each worker a unique ``initial_seed``; this function forwards
    that seed to NumPy and Python's ``random`` module, which PyTorch does not
    seed automatically. Passed as ``worker_init_fn`` to every DataLoader.

    Args:
        worker_id: Worker process index. Not used directly — PyTorch already
            incorporates it into ``torch.initial_seed()``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------
# 2. Data Discovery
# ----------------------------------------------------


def get_data_list(root_dir: str, tabular_csv: Optional[str] = None) -> List[Dict]:
    """
    Discover NIfTI files under a dataset root and return them as record dicts.

    Expected layout::

        root_dir/
            0/   ← negative cases (no aneurysm)
            1/   ← positive cases (aneurysm present)

    Each record contains ``"image"`` (absolute path), ``"label"`` (0 or 1),
    and ``"path"`` (filename). When ``tabular_csv`` is provided, a ``"tabular"``
    key is added with a float32 array of ``[age/100, gender]``.

    Args:
        root_dir: Path to the dataset root containing ``0/`` and ``1/`` subdirs.
        tabular_csv: Optional CSV with columns ``exam``, ``Age``, ``gender``
            (matches ``data_engine/dataset/classes.csv``). Rows where ``Age``
            or ``gender`` are blank are skipped.

    Returns:
        List of record dicts, one per NIfTI file.

    Raises:
        KeyError: If a NIfTI file has no corresponding row in the tabular CSV.
    """
    tabular_data: Dict[str, np.ndarray] = {}
    if tabular_csv is not None:
        with open(tabular_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows where age or gender are missing (e.g. excluded cases in classes.csv,
                # or before gender data has been populated).
                if not row.get("Age") or not row.get("gender"):
                    continue
                # Age normalised to [0, 1] by dividing by 100 (known patient age range).
                # Gender encoded as 0.0 / 1.0.
                tabular_data[row["exam"]] = np.array(
                    [float(row["Age"]) / 100.0, float(row["gender"])],
                    dtype=np.float32,
                )

    data = []
    for label in [0, 1]:
        class_folder = os.path.join(root_dir, str(label))
        if not os.path.exists(class_folder):
            continue
        for f in os.listdir(class_folder):
            if f.endswith((".nii", ".nii.gz")):
                record: Dict = {
                    "image": os.path.join(class_folder, f),
                    "label": label,
                    "path": f,
                }
                if tabular_csv is not None:
                    case_id = f.replace(".nii.gz", "").replace(".nii", "")
                    if case_id not in tabular_data:
                        raise KeyError(
                            f"Case '{case_id}' (file '{f}') not found in tabular CSV "
                            f"'{tabular_csv}'. Every NIfTI file must have a matching row."
                        )
                    record["tabular"] = tabular_data[case_id]
                data.append(record)
    return data


# ----------------------------------------------------
# 3. Class Imbalance Weighting
# ----------------------------------------------------


def get_class_weights(data: List[Dict]) -> Optional[torch.Tensor]:
    """
    Compute inverse-frequency class weights for use with ``CrossEntropyLoss``.

    Weight for class c = ``total_samples / count_c``, so the minority class
    receives a proportionally larger gradient contribution. Must be called on
    the training fold only to avoid label leakage from held-out sets.

    Args:
        data: Training fold record dicts, each with a ``"label"`` key (0 or 1).

    Returns:
        Float tensor of shape ``[2]`` with ``[weight_class_0, weight_class_1]``,
        or ``None`` if ``data`` is empty.
    """
    if not data:
        return None
    labels = [d["label"] for d in data]
    total_samples = len(labels)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = [
        total_samples / count if count > 0 else 0.0
        for count in class_counts
    ]
    return torch.tensor(class_weights, dtype=torch.float)


# ----------------------------------------------------
# 4. MONAI Transforms
# ----------------------------------------------------


def get_transforms(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    augment: bool = False,
    use_tabular: bool = False,
) -> Compose:
    """
    Build the MONAI dictionary-based transform pipeline for 3D NIfTI volumes.

    Baseline transforms (always applied): ``LoadImaged`` → ``EnsureChannelFirstd``
    → ``Resized`` → ``ScaleIntensityd`` → ``EnsureTyped``.

    Augmentation transforms (``augment=True``, training only): three axis-wise
    ``RandFlipd``, ``RandRotate90d`` (0/90/180/270° in axial plane), ``RandRotated``
    (±15°), ``RandScaleIntensityd``, ``RandZoomd`` (90–110%), ``RandGaussianNoised``
    (std=0.05 on [0,1]-scaled data).

    Args:
        spatial_size: Target ``(H, W, D)`` shape for ``Resized``.
        augment: If ``True``, appends stochastic augmentation transforms.
        use_tabular: If ``True``, includes ``"tabular"`` in ``EnsureTyped``.

    Returns:
        A ``Compose`` object containing the full transform chain.
    """
    keys = ["image"]
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=spatial_size),
        ScaleIntensityd(keys=keys),
    ]
    if augment:
        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandRotated(keys=keys, range_x=0.26, range_y=0.26, range_z=0.26, prob=0.5),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=0.5),
            RandZoomd(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.3, keep_size=True),
            RandGaussianNoised(keys=keys, prob=0.2, mean=0.0, std=0.05),
        ])
    ensure_keys = ["image", "tabular"] if use_tabular else ["image"]
    transforms.append(EnsureTyped(keys=ensure_keys))
    return Compose(transforms)


# ----------------------------------------------------
# 5. Data Splitting  (pure Python — no MONAI/DataLoader)
# ----------------------------------------------------


def split_data(
    all_data: List[Dict],
    n_splits: int,
    test_size: float,
    seed: int = 42,
    use_kfold: bool = True,
    val_split_ratio: float = 0.30,
) -> Generator[Tuple[int, List[Dict], List[Dict], List[Dict]], None, None]:
    """
    Partition ``all_data`` into stratified train / val / test subsets and yield them.

    First splits off a permanent hold-out test set (``test_size`` fraction),
    then divides the remaining development set into train/val. When
    ``use_kfold=True``, yields one tuple per stratified k-fold split; when
    ``False``, yields a single tuple using ``val_split_ratio``.

    The test set is identical across all folds. Stratification preserves the
    class ratio in every split to prevent minority-class concentration by chance.

    Args:
        all_data: Full list of record dicts from ``get_data_list``.
        n_splits: Number of folds for k-fold CV. Ignored when ``use_kfold=False``.
        test_size: Fraction of ``all_data`` to reserve as the hold-out test set.
        seed: Random seed for reproducible splits.
        use_kfold: If ``True``, yield one tuple per fold. If ``False``, yield a
            single tuple with a simple stratified train/val split.
        val_split_ratio: Validation fraction of the dev set when ``use_kfold=False``.

    Yields:
        Tuples of ``(fold_idx, train_files, val_files, test_files)``.
    """
    labels = [x["label"] for x in all_data]

    # Stratification requires at least 2 samples per class in every partition.
    # Fall back to non-stratified splitting when any class is too small.
    can_stratify = min(labels.count(c) for c in set(labels)) >= 2
    if not can_stratify:
        print(
            "  Warning [split_data]: too few samples per class for stratified splitting "
            f"(class counts: { {c: labels.count(c) for c in set(labels)} }). "
            "Falling back to non-stratified random split."
        )

    stratified_dev_indices, test_indices = train_test_split(
        range(len(all_data)),
        test_size=test_size,
        shuffle=True,
        stratify=labels if can_stratify else None,
        random_state=seed,
    )
    dev_data = [all_data[i] for i in stratified_dev_indices]
    test_files = [all_data[i] for i in test_indices]
    dev_labels = [d["label"] for d in dev_data]

    can_stratify_dev = min(dev_labels.count(c) for c in set(dev_labels)) >= 2 if dev_labels else False

    if not use_kfold:
        train_idx, val_idx = train_test_split(
            range(len(dev_data)),
            test_size=val_split_ratio,
            shuffle=True,
            stratify=dev_labels if can_stratify_dev else None,
            random_state=seed,
        )
        yield 0, [dev_data[i] for i in train_idx], [dev_data[i] for i in val_idx], test_files
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_data, dev_labels)):
        train_files = [dev_data[i] for i in train_idx]
        val_files = [dev_data[i] for i in val_idx]
        yield fold_idx, train_files, val_files, test_files


# ----------------------------------------------------
# 6. DataLoader Assembly
# ----------------------------------------------------


def build_dataloaders(
    train_files: List[Dict],
    val_files: List[Dict],
    test_files: List[Dict],
    batch_size: int,
    val_batch_size: int,
    seed: int = 42,
    oversample: bool = False,
    use_tabular: bool = False,
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    cache_rate: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap train, val, and test record lists in MONAI Datasets and PyTorch DataLoaders.

    - Training loader uses augmentation; val/test loaders do not.
    - When ``oversample=True``, a ``WeightedRandomSampler`` balances class frequency
      per epoch by drawing with replacement (``shuffle`` is disabled when a sampler
      is active).
    - When ``cache_rate > 0``, ``CacheDataset`` caches deterministic transforms in RAM,
      eliminating repeated NIfTI I/O from epoch 2 onwards.
    - Each loader receives a seeded ``generator`` and ``seed_worker`` as
      ``worker_init_fn`` for full reproducibility.

    Args:
        train_files: Training fold record dicts.
        val_files: Validation fold record dicts.
        test_files: Hold-out test set record dicts.
        batch_size: Number of volumes per training batch.
        val_batch_size: Number of volumes per validation/test batch.
        seed: Random seed for DataLoader generators and worker init.
        oversample: If ``True``, attach a ``WeightedRandomSampler`` to the training loader.
        use_tabular: If ``True``, convert the ``"tabular"`` key to a tensor.
        spatial_size: Target ``(H, W, D)`` shape for the ``Resized`` transform.
        cache_rate: Fraction of the dataset to cache in RAM (0.0–1.0).

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    train_transform = get_transforms(spatial_size=spatial_size, augment=True, use_tabular=use_tabular)
    eval_transform = get_transforms(spatial_size=spatial_size, augment=False, use_tabular=use_tabular)

    sampler = None
    if oversample:
        fold_labels = [x["label"] for x in train_files]
        fold_class_counts = np.bincount(fold_labels)
        per_class_sample_weights = 1.0 / fold_class_counts
        sample_weights = torch.from_numpy(
            np.array([per_class_sample_weights[x["label"]] for x in train_files])
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    if cache_rate > 0.0:
        train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=cache_rate, num_workers=4, progress=False)
        val_ds = CacheDataset(data=val_files, transform=eval_transform, cache_rate=cache_rate, num_workers=2, progress=False)
        test_ds = CacheDataset(data=test_files, transform=eval_transform, cache_rate=cache_rate, num_workers=2, progress=False)
    else:
        train_ds = Dataset(data=train_files, transform=train_transform)
        val_ds = Dataset(data=val_files, transform=eval_transform)
        test_ds = Dataset(data=test_files, transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    return train_loader, val_loader, test_loader
