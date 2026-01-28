import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import torchvision.transforms as transforms
import os
import random
from typing import Tuple, Union, Any, Callable, Optional, List, Dict
import glob
import nibabel as nib
import numpy as np
from collections import Counter

# ----------------------------------------------------
# 1. Define NIfTI Transforms (Augmented vs. Standard)
# ----------------------------------------------------

class NiftiAugment3D(object):
    """
    Applies 3D augmentations (Flip, Gaussian Noise, Rot90) to a NIfTI Tensor.
    Assumes input tensor is (D, H, W) or (C, D, H, W) where C is the 0th dim.
    """
    def __init__(self, flip_p=0.5, noise_p=0.3, rot_p=0.5):
        self.flip_p = flip_p
        self.noise_p = noise_p
        self.rot_p = rot_p

    def __call__(self, volume_tensor: torch.Tensor) -> torch.Tensor:
        # Determine if a channel dimension is present (ndim=4, e.g., 1, 128, 128, 128)
        is_4d = (volume_tensor.ndim == 4)
        if is_4d:
            volume = volume_tensor.squeeze(0) # (D, H, W)
            W_AXIS = 2
            H_AXIS = 1
        else:
            volume = volume_tensor # (D, H, W)
            W_AXIS = 2
            H_AXIS = 1

        # 1. Random Flip (Left-Right) - typically W-axis
        if np.random.rand() < self.flip_p:
            volume = torch.flip(volume, dims=[W_AXIS])

        # 2. Random Gaussian Noise
        if np.random.rand() < self.noise_p:
            data_range = volume.max() - volume.min()
            # Note: 0.01 * data_range is the same relative std_dev as in pipeline1.py
            std_dev = 0.01 * data_range
            noise = torch.randn_like(volume) * std_dev
            volume = volume + noise
            volume = torch.clamp(volume, min=0.0)

        # 3. Random Rot90 (90, 180, 270 degrees) in the axial plane (H, W)
        if np.random.rand() < self.rot_p:
            k = np.random.randint(1, 4) # 1, 2, or 3 times 90 degrees
            volume = torch.rot90(volume, k, dims=[H_AXIS, W_AXIS])

        # Add the channel dimension back if it was present initially
        if is_4d:
            return volume.unsqueeze(0) # (1, D, H, W)
        else:
            return volume # (D, H, W)


# --- FIX 1: Removed transforms.ToTensor() to avoid implicit 1/255 scaling. ---
# The volume conversion to Tensor and 0-1 scaling are now handled in NiftiFolder.__getitem__.
# These transforms now only handle channel addition and augmentation.

train_transform = transforms.Compose([
    # This Lambda layer adds the Channel dimension (C, D, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(0)),
    NiftiAugment3D(flip_p=0.5, noise_p=0.3, rot_p=0.5)
])

standard_transform = transforms.Compose([
    # This Lambda layer adds the Channel dimension (C, D, H, W)
    transforms.Lambda(lambda x: x.unsqueeze(0)),
])


# ----------------------------------------------------
# 2. Custom NIfTI Dataset Class (NiftiFolder and NEW BalancedSubset)
# ----------------------------------------------------

class NiftiFolder(Dataset):
    """
    A custom Dataset to load 3D NIfTI files, organized like ImageFolder,
    but designed to return the file path as the third item.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root)
        self.samples = self._make_dataset(root, self.class_to_idx)

        if not self.samples:
            raise RuntimeError(f"Found 0 NIfTI files in: {root}")

    def _find_classes(self, dir: str) -> Tuple[List[str], dict]:
        """Finds the class folder names and returns a list of classes and a dict of class_to_idx."""
        classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir: str, class_to_idx: dict) -> List[Tuple[str, int]]:
        """Creates a list of (file_path, class_index) tuples."""
        samples = []
        nifti_extensions = ['*.nii', '*.nii.gz']

        for class_name, class_index in class_to_idx.items():
            class_dir = os.path.join(dir, class_name)
            for ext in nifti_extensions:
                for path in glob.glob(os.path.join(class_dir, ext)):
                    samples.append((path, class_index))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any, str]:
        """Loads and returns the 3D scan, its label, and file path."""
        path, target = self.samples[index]

        # 1. Load the NIfTI file
        img = nib.load(path)
        volume = img.get_fdata(dtype=np.float32)

        # 2. Basic normalization (0-1 scaling) - MATCHES pipeline1.py
        if volume.max() > 0:
             volume = volume / volume.max()

        # --- FIX 2: Convert to PyTorch Tensor here, before applying any composed transforms. ---
        # This ensures the tensor is float32 and prevents transforms.ToTensor() from scaling.
        volume = torch.from_numpy(volume).float()

        # 3. Apply the transform (now only contains channel addition/augmentation)
        if self.transform is not None:
            volume = self.transform(volume)

        return volume, target, path

class BalancedSubset(Dataset):
    """
    A custom Dataset wrapper that takes a PyTorch Subset and oversamples the
    minority class to match the majority class size.
    Since augmentation is applied in the base dataset, each sampled minority
    instance will be uniquely augmented.
    """
    def __init__(self, subset: Subset):
        self.subset = subset

        # 1. Get original indices and labels
        original_indices = subset.indices
        # Must access the base dataset of the subset to get the label from the sample list
        original_labels = [self.subset.dataset.samples[i][1] for i in original_indices]

        # 2. Count class distribution
        class_counts = Counter(original_labels)
        self.majority_class = max(class_counts, key=class_counts.get)
        self.minority_class = min(class_counts, key=class_counts.get)

        self.majority_size = class_counts[self.majority_class]
        self.minority_size = class_counts[self.minority_class]

        # 3. Separate indices for majority and minority
        majority_indices = [idx for idx, label in zip(original_indices, original_labels) if label == self.majority_class]
        minority_indices = [idx for idx, label in zip(original_indices, original_labels) if label == self.minority_class]

        # 4. Perform Oversampling on minority indices
        # Calculate how many times the minority set needs to be repeated (with replacement)
        num_repeats = self.majority_size // self.minority_size
        remainder = self.majority_size % self.minority_size

        # Repeat and add the remainder
        oversampled_minority_indices = (minority_indices * num_repeats) + random.sample(minority_indices, remainder)

        # 5. Combine and shuffle the final balanced list of indices
        self.final_indices = majority_indices + oversampled_minority_indices
        random.shuffle(self.final_indices)

        print(f"   [Balancing]: Majority ({self.majority_class}): {self.majority_size} | Minority ({self.minority_class}): {self.minority_size}")
        print(f"   [Balancing]: Final set size: {len(self.final_indices)} (Balanced: {self.majority_size} per class)")

    def __len__(self) -> int:
        return len(self.final_indices)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, str]:
        # The index in final_indices points to the index in the original base dataset (NiftiFolder)
        original_base_index = self.final_indices[idx]
        return self.subset.dataset.__getitem__(original_base_index)


# ----------------------------------------------------
# 3. Helper Functions
# ----------------------------------------------------

def set_seed(seed: int):
    """Sets the random seed for reproducibility across multiple libraries."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_full_dataset(root_path: str) -> NiftiFolder:
    """Loads all NIfTI files using the NiftiFolder."""
    if not os.path.exists(root_path) or not os.listdir(root_path):
        print(f"Error: Dataset root path '{root_path}' does not exist or is empty.")
        raise FileNotFoundError(f"Root path {root_path} not found or is empty.")

    print(f"Loading full dataset from: {root_path}")
    full_dataset = NiftiFolder(root=root_path, transform=None)
    print(f"Detected Classes: {full_dataset.classes}")
    return full_dataset

# ----------------------------------------------------
# 4. Main Data Loading Function (MODIFIED for Oversampling)
# ----------------------------------------------------

def get_data_loaders(
    root_path: str,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    oversample_train: bool = False # <--- NEW OPTIONAL FLAG
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Performs the 3-way split (Train/Validation/Test) and returns the respective DataLoaders.
    Optionally oversamples the training set's minority class.
    """
    # 1. Set seed
    set_seed(random_seed)

    # 2. Load the full dataset (NiftiFolder instance)
    full_raw_dataset = load_full_dataset(root_path)

    # 3. Calculate split sizes
    TOTAL_SIZE = len(full_raw_dataset)
    train_size = int(train_ratio * TOTAL_SIZE)
    val_size = int(val_ratio * TOTAL_SIZE)
    test_size = TOTAL_SIZE - train_size - val_size

    # Perform the split to get indices (Uses torch.Generator().manual_seed(random_seed) - MATCHES pipeline1.py)
    generator = torch.Generator().manual_seed(random_seed)
    all_indices = range(TOTAL_SIZE)
    train_indices, val_indices, test_indices = random_split(
        all_indices,
        [train_size, val_size, test_size],
        generator=generator
    )

    # 4. Apply Transforms and Create Subsets
    # Use the base NiftiFolder with the new, simpler transforms
    train_base = NiftiFolder(root=root_path, transform=train_transform)
    val_base = NiftiFolder(root=root_path, transform=standard_transform)
    test_base = NiftiFolder(root=root_path, transform=standard_transform)

    # Create initial subsets (always needed)
    train_subset = Subset(train_base, train_indices.indices)
    val_dataset = Subset(val_base, val_indices.indices)
    test_dataset = Subset(test_base, test_indices.indices)

    # 5. Apply Oversampling if requested
    if oversample_train:
        print("\nApplying Oversampling to Training Set...")
        train_dataset = BalancedSubset(train_subset) # Wrap the original train_subset
    else:
        train_dataset = train_subset

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("-" * 50)
    print("Data Loading Complete.")
    print(f"Train set size: {len(train_dataset)} (Augmented Transform {'+ Oversampled' if oversample_train else ''})")
    print(f"Valid set size: {len(val_dataset)} (Standard Transform)")
    print(f"Test set size: {len(test_dataset)} (Standard Transform)")
    print("-" * 50)

    return train_loader, val_loader, test_loader