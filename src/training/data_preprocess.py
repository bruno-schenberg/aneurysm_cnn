import os
import torch
import numpy as np
import random
from typing import List, Dict, Optional
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityd, RandFlipd, RandRotate90d, RandGaussianNoised,
    EnsureTyped,
)
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedKFold, train_test_split

# ----------------------------------------------------
# 1. Global Reproducibility
# ----------------------------------------------------

def set_seed(seed: int = 42):
    """Sets all seeds and MONAI determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # MONAI-specific deterministic backend
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Ensures DataLoader workers maintain determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ----------------------------------------------------
# 2. Data Discovery & Splitting
# ----------------------------------------------------

def get_data_list(root_dir: str) -> List[Dict]:
    """
    Scans folders '0' and '1' and returns a list of dictionaries.
    Each dict contains path and label to prevent losing track of files.
    """
    data = []
    for label in [0, 1]:
        class_folder = os.path.join(root_dir, str(label))
        if not os.path.exists(class_folder):
            continue
        for f in os.listdir(class_folder):
            if f.endswith(('.nii', '.nii.gz')):
                data.append({
                    "image": os.path.join(class_folder, f),
                    "label": label,
                    "path": f  # Store filename for later analysis
                })
    return data

# ----------------------------------------------------
# 3. Data Analysis & Weighting
# ----------------------------------------------------

def get_class_counts(data_dir: str, classes: list) -> dict:
    """
    Scans the data directory to count files in each class subfolder.
    This makes class weight calculation dynamic and dataset-agnostic.
    """
    counts = {}
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for class_name in classes:
        class_path = os.path.join(data_dir, str(class_name))
        # Count files if the directory exists, otherwise count is 0
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]) if os.path.isdir(class_path) else 0
        counts[str(class_name)] = count
    return counts

def get_class_weights(data_dir: str, classes: list) -> Optional[torch.Tensor]:
    """
    Calculates class weights for a weighted loss function.
    The weight is the inverse of the class frequency.
    """
    class_counts = get_class_counts(data_dir, classes)
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return None
    counts_per_class = [class_counts[c] for c in classes]
    class_weights_list = [total_samples / count if count > 0 else 0 for count in counts_per_class]
    return torch.tensor(class_weights_list, dtype=torch.float)

# ----------------------------------------------------
# 3. MONAI Transforms
# ----------------------------------------------------

def get_transforms(spatial_size=(128, 128, 128), augment=False):
    """
    Defines the dictionary-based transform pipeline.
    Augmentations are ONLY applied if augment=True.
    """
    keys = ["image"]
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=spatial_size),
        ScaleIntensityd(keys=keys), # Robust 0-1 scaling
    ]

    if augment:
        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            RandGaussianNoised(keys=keys, prob=0.2, mean=0.0, std=0.01)
        ])
    
    transforms.append(EnsureTyped(keys=keys))
    return Compose(transforms)

# ----------------------------------------------------
# 4. The Balanced K-Fold Generator
# ----------------------------------------------------

def get_folds(data_dir: str, n_splits: int, batch_size: int, val_batch_size: int, test_size: float, seed: int = 42, oversample: bool = False):
    """
    Performs a stratified split of the data into a development set and a hold-out test set.
    Then, it creates a generator that yields (train_loader, val_loader) for each K-fold
    split of the development set, along with a single, consistent test_loader.
    Includes oversampling logic inside each training fold to prevent data leakage.
    """
    set_seed(seed)
    all_data = get_data_list(data_dir)
    labels = [x['label'] for x in all_data]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # --- New: Initial split into development (train+val) and test sets ---
    dev_indices, test_indices = train_test_split(
        range(len(all_data)),
        test_size=test_size,
        shuffle=True,
        stratify=labels,
        random_state=seed
    )
    dev_data = [all_data[i] for i in dev_indices]
    test_files = [all_data[i] for i in test_indices]
    dev_labels = [d['label'] for d in dev_data]
    # ---

    train_trans = get_transforms(augment=True)
    val_trans = get_transforms(augment=False)

    # The K-Fold split is now performed only on the development data
    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_data, dev_labels)):
        train_files = [dev_data[i] for i in train_idx]
        val_files = [dev_data[i] for i in val_idx]

        # Handle Oversampling for this specific fold
        sampler = None
        if oversample:
            fold_labels = [x['label'] for x in train_files]
            class_counts = np.bincount(fold_labels)
            weights = 1. / class_counts
            samples_weights = torch.from_numpy(np.array([weights[x['label']] for x in train_files]))
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=samples_weights, num_samples=len(samples_weights), replacement=True
            )

        # Use MONAI's standard Dataset to load data on-the-fly and avoid high memory usage.
        train_ds = Dataset(data=train_files, transform=train_trans)
        val_ds = Dataset(data=val_files, transform=val_trans)
        test_ds = Dataset(data=test_files, transform=val_trans) # Use non-augmenting transforms

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, 
            num_workers=4, worker_init_fn=seed_worker, shuffle=(sampler is None)
        )
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=2)
        # The test loader is the same for every fold
        test_loader = DataLoader(test_ds, batch_size=val_batch_size, num_workers=2)

        yield fold, train_loader, val_loader, test_loader