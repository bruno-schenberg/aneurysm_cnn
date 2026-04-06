"""
data_preprocess.py

Handles all data preparation for the training engine: NIfTI file discovery,
stratified data splitting, class-imbalance weighting, MONAI transform construction,
and DataLoader assembly. This module is the boundary between raw data on disk and
the PyTorch training loop.
"""

import os
import random
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    Resized,
    ScaleIntensityd,
)
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedKFold, train_test_split

# ----------------------------------------------------
# 1. Global Reproducibility
# ----------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Sets all random seeds and enables MONAI deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seeds each DataLoader worker to maintain per-worker determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------
# 2. Data Discovery
# ----------------------------------------------------


def get_data_list(root_dir: str) -> List[Dict]:
    """
    Discovers NIfTI files in class subdirectories (0/ and 1/) and returns
    a list of DataRecord dicts with image path, label, and filename.
    """
    data = []
    for label in [0, 1]:
        class_folder = os.path.join(root_dir, str(label))
        if not os.path.exists(class_folder):
            continue
        for f in os.listdir(class_folder):
            if f.endswith((".nii", ".nii.gz")):
                data.append({
                    "image": os.path.join(class_folder, f),
                    "label": label,
                    "path": f,
                })
    return data


# ----------------------------------------------------
# 3. Class Imbalance Weighting
# ----------------------------------------------------


def get_class_weights(data: List[Dict]) -> Optional[torch.Tensor]:
    """
    Computes inverse-frequency class weights from a list of DataRecords.
    Weights are proportional to 1 / class_count, suitable for passing to
    CrossEntropyLoss to counteract class imbalance.

    Must be called with the training fold's data only (not the full dataset)
    to prevent label leakage into the loss function.

    Args:
        data: List of DataRecord dicts, each containing a 'label' key (0 or 1).

    Returns:
        Tensor of shape [2] with [weight_class_0, weight_class_1],
        or None if the data list is empty.
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
) -> Compose:
    """
    Builds the dictionary-based MONAI transform pipeline.
    Augmentations (random flip, rotate, Gaussian noise) are applied only when augment=True.
    """
    keys = ["image"]
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=spatial_size),
        ScaleIntensityd(keys=keys),  # Robust 0–1 scaling
    ]
    if augment:
        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotated(keys=keys, range_x=0.26, range_y=0.26, range_z=0.26, prob=0.5),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=0.5),
            RandZoomd(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.3, keep_size=True),
            RandGaussianNoised(keys=keys, prob=0.2, mean=0.0, std=0.01),
        ])
    transforms.append(EnsureTyped(keys=keys))
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
    Partitions all_data into a stratified hold-out test set and a development
    set, then yields (fold_idx, train_files, val_files, test_files).

    When use_kfold=True: yields one tuple per stratified k-fold split.
    When use_kfold=False: yields a single tuple using a stratified
    train/val split with val_split_ratio as the validation fraction.

    This function is intentionally free of MONAI and PyTorch to keep it
    unit-testable with synthetic in-memory data on any machine.
    """
    labels = [x["label"] for x in all_data]

    stratified_dev_indices, test_indices = train_test_split(
        range(len(all_data)),
        test_size=test_size,
        shuffle=True,
        stratify=labels,
        random_state=seed,
    )
    dev_data = [all_data[i] for i in stratified_dev_indices]
    test_files = [all_data[i] for i in test_indices]
    dev_labels = [d["label"] for d in dev_data]

    if not use_kfold:
        train_idx, val_idx = train_test_split(
            range(len(dev_data)),
            test_size=val_split_ratio,
            shuffle=True,
            stratify=dev_labels,
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wraps train, val, and test file lists in MONAI Datasets and PyTorch DataLoaders.
    Applies training augmentations to the training set only.
    Oversampling (WeightedRandomSampler) is applied to the training loader when requested.

    Both worker_init_fn and generator are set on every DataLoader for full reproducibility.
    """
    train_transform = get_transforms(augment=True)
    eval_transform = get_transforms(augment=False)

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


# ----------------------------------------------------
# 7. Backwards-Compatible Fold Generator
# ----------------------------------------------------


def get_folds(
    data_dir: str,
    n_splits: int,
    batch_size: int,
    val_batch_size: int,
    test_size: float,
    seed: int = 42,
    oversample: bool = False,
) -> Generator:
    """
    Thin wrapper around split_data + build_dataloaders for backwards compatibility.
    Yields (fold_idx, train_loader, val_loader, test_loader) for each fold.
    """
    set_seed(seed)
    all_data = get_data_list(data_dir)
    for fold_idx, train_files, val_files, test_files in split_data(
        all_data, n_splits, test_size, seed
    ):
        train_loader, val_loader, test_loader = build_dataloaders(
            train_files, val_files, test_files,
            batch_size, val_batch_size, seed, oversample,
        )
        yield fold_idx, train_loader, val_loader, test_loader
