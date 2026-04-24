"""
data_preprocess.py

Handles all data preparation for the training engine: NIfTI file discovery,
stratified data splitting, class-imbalance weighting, MONAI transform construction,
and DataLoader assembly. This module is the boundary between raw data on disk and
the PyTorch training loop.

## Responsibilities

This module does not produce or modify any NIfTI files — that work is done
upstream by the data engine. By the time this module runs, every volume is
already a 128×128×128 float32 NIfTI living in one of two class subdirectories
(``0/`` for negative, ``1/`` for positive). This module's job is to:

  1. Discover those files and attach labels (``get_data_list``).
  2. Split them reproducibly into train / val / test subsets (``split_data``).
  3. Compute per-class weights to counteract imbalance (``get_class_weights``).
  4. Build the MONAI transform pipeline — loading, intensity scaling, and
     optional augmentation (``get_transforms``).
  5. Wrap everything in PyTorch DataLoaders with full reproducibility
     guarantees (``build_dataloaders``).

## Class imbalance: two strategies

Aneurysm datasets are typically imbalanced — there are more negative cases
(no aneurysm) than positive ones. Two complementary strategies are available
and can be selected per-experiment:

  - **Weighted loss** (``get_class_weights``): passes per-class weights to
    ``CrossEntropyLoss`` so that mistakes on the rare class are penalised more
    heavily. The model still sees the natural class distribution in training
    batches, but the gradient signal is amplified for the minority class.

  - **Oversampling** (``WeightedRandomSampler`` in ``build_dataloaders``):
    draws training samples with replacement so that minority-class examples
    appear as often as majority-class examples in each epoch. The loss function
    is not modified — every example contributes equally once sampled.

Both strategies aim for the same goal (preventing the model from learning to
always predict the majority class) via different mechanisms. They can be
combined or used in isolation; the choice is an experimental variable.

## Reproducibility

Three separate random number streams are seeded here:

  - Python's ``random`` module — used by some MONAI transforms internally.
  - NumPy — used by scikit-learn splits and oversampling weight computation.
  - PyTorch (CPU and all CUDA devices) — governs all model and DataLoader
    stochasticity.

MONAI's ``set_determinism`` seeds its own internal operations. Setting
``cudnn.deterministic = True`` and disabling ``cudnn.benchmark`` ensures that
CUDA convolution algorithms are chosen deterministically rather than by runtime
auto-tuning, at a small cost in throughput.

DataLoader workers are separate processes that inherit a copy of the parent's
random state at spawn time, which means they would all produce identical random
sequences unless explicitly re-seeded. ``seed_worker`` (passed as
``worker_init_fn``) derives a unique seed for each worker from the PyTorch
worker seed, which is itself derived from the DataLoader's ``generator``.
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

    **Why string → tuple at the config boundary?**
    JSON has no tuple type, so ``INPUT_RESOLUTION`` is stored as a
    human-readable string (e.g. ``"256x256x128"``) in ``experiments.json``.
    Converting to a typed tuple at the config-validation boundary means the
    rest of the pipeline — transforms, DataLoaders, orchestrator — always
    receives a ``Tuple[int, int, int]`` and never has to handle string parsing.
    This mirrors how ``data_path_key`` is resolved to a filesystem path in
    ``train_models.py`` before reaching the pipeline.

    **Why ``VALID_RESOLUTIONS`` is a dict?**
    The dict serves two purposes: it is the exhaustive whitelist of permitted
    values (an unlisted string is immediately rejected) and it performs the
    parsing in a single lookup without any string-splitting arithmetic.
    Centralising this in one dict means adding a new resolution requires
    only a new entry here — validation and parsing stay in sync automatically.

    Args:
        resolution: A resolution string that must be a key in
            ``VALID_RESOLUTIONS`` (e.g. ``"128x128x128"``).

    Returns:
        A ``(H, W, D)`` integer tuple suitable for passing to MONAI's
        ``Resized`` transform as ``spatial_size``.

    Raises:
        ValueError: If ``resolution`` is not a key in ``VALID_RESOLUTIONS``,
            with a message that names the invalid value and lists valid options.
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
    Seeds all random number generators and enables MONAI/cuDNN deterministic mode.

    Five separate sources of randomness are seeded because they are independent
    libraries with independent internal states:

      - ``random.seed``: Python's built-in RNG, used implicitly by some MONAI
        transforms and by Python's own ``shuffle``/``choice`` calls.
      - ``np.random.seed``: NumPy's legacy global RNG, used by scikit-learn
        (splits, stratification) and by NumPy operations in MONAI.
      - ``torch.manual_seed``: seeds the CPU RNG for all torch operations
        including weight initialisation and stochastic transforms.
      - ``torch.cuda.manual_seed_all``: seeds the RNG on every available GPU.
        Without this, GPU operations use a different (non-deterministic) seed.
      - ``set_determinism``: MONAI's wrapper that calls ``torch.manual_seed``,
        seeds NumPy, and configures MONAI-internal random states.

    The two cuDNN flags handle a separate source of non-determinism at the
    CUDA kernel level. When ``benchmark=True`` (the default), CUDA auto-tunes
    which convolution algorithm is fastest for the current input shape, and the
    choice can vary between runs. Setting ``deterministic=True`` and
    ``benchmark=False`` disables this tuning and forces reproducible algorithm
    selection, at a small performance cost.

    Args:
        seed: Integer seed applied to all RNG sources. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seeds the random state of each DataLoader worker process.

    PyTorch DataLoaders spawn ``num_workers`` separate child processes. Each
    worker inherits a copy of the parent's random state at spawn time, which
    means all workers would generate identical random sequences — defeating the
    purpose of independent data augmentation across parallel workers.

    PyTorch solves this by giving each worker a unique ``initial_seed``:
    ``base_seed + worker_id``. This function reads that per-worker seed via
    ``torch.initial_seed()`` and applies it to NumPy and Python's ``random``
    module, which are not automatically seeded by PyTorch.

    The modulo ``% 2**32`` is required because ``torch.initial_seed()`` returns
    a 64-bit integer, but ``np.random.seed`` only accepts 32-bit values.

    This function is passed as ``worker_init_fn`` to every DataLoader, alongside
    an explicit ``generator`` argument that controls the base seed. Together they
    give fully reproducible, per-worker augmentation sequences.

    Args:
        worker_id: Integer index of the worker process (0-indexed). Not used
            directly — PyTorch already incorporates it into ``initial_seed()``.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------
# 2. Data Discovery
# ----------------------------------------------------


def get_data_list(root_dir: str, tabular_csv: Optional[str] = None) -> List[Dict]:
    """
    Discovers NIfTI files under a dataset root and returns them as a flat list
    of record dicts ready for use with MONAI's dictionary-based transforms.

    The expected directory layout mirrors what the data engine produces:

        root_dir/
            0/          ← negative cases (no aneurysm)
                BP001.nii.gz
                BP002.nii.gz
                ...
            1/          ← positive cases (aneurysm present)
                BP003.nii.gz
                ...

    This two-class folder convention is a standard pattern for binary
    classification datasets. It keeps the label implicit in the folder name,
    avoiding the need for a separate CSV manifest and making the dataset
    self-describing on disk.

    If one class folder is missing entirely (e.g. during testing with a
    partial dataset) the function silently skips it rather than raising an
    error, so the caller can still operate on whatever is available.

    Each record dict contains:
      - ``"image"``: absolute path to the NIfTI file, used by ``LoadImaged``.
      - ``"label"``: integer 0 or 1, used by the training loop loss function.
      - ``"path"``: the filename only, used for logging and result bookkeeping.

    When ``tabular_csv`` is provided, each record also gets a ``"tabular"`` key
    containing a float32 numpy array of ``[age/100, gender]`` for that case.
    The CSV must have columns: ``case_id``, ``age``, ``gender`` — where
    ``case_id`` matches the NIfTI filename without its extension
    (e.g. ``'BP001'`` for ``'BP001.nii.gz'``).

    Args:
        root_dir: Path to the dataset root containing ``0/`` and ``1/``
            subdirectories.
        tabular_csv: Optional path to a CSV with patient metadata columns
            ``case_id``, ``age``, ``gender``.

    Returns:
        List of record dicts, one per NIfTI file, in filesystem order.

    Raises:
        KeyError: If a NIfTI file has no corresponding row in the tabular CSV.
    """
    tabular_data: Dict[str, np.ndarray] = {}
    if tabular_csv is not None:
        with open(tabular_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Age normalised to [0, 1] by dividing by 100 (known patient age range).
                # Gender encoded as 0.0 / 1.0.
                tabular_data[row["case_id"]] = np.array(
                    [float(row["age"]) / 100.0, float(row["gender"])],
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
    Computes inverse-frequency class weights for use with ``CrossEntropyLoss``.

    **Why weight the loss?**
    In a dataset where 80% of cases are negative and 20% positive, a model can
    achieve 80% accuracy by always predicting "negative". Standard cross-entropy
    loss treats every sample equally, so the gradient is dominated by the
    majority class and the model quickly learns this degenerate solution.
    Inverse-frequency weighting amplifies the gradient contribution of the
    minority class so that misclassifying a positive case is penalised more
    heavily than misclassifying a negative case.

    **How the weights are computed:**
    For each class c, ``weight_c = total_samples / count_c``. This makes the
    effective contribution of each class equal: a class with half as many
    samples gets twice the per-sample weight, so the total weighted gradient
    from each class is balanced.

    **Why call this on the training fold only?**
    Class weights must be computed from the training fold, not the full dataset.
    Computing them from the full dataset would incorporate knowledge of the
    validation and test set label distributions, which constitutes label leakage
    — the loss function would be implicitly tuned to the held-out sets. Using
    the training fold alone keeps the weighting scheme honest.

    The ``minlength=2`` argument to ``np.bincount`` ensures the output always
    has length 2 even if one class is absent from the data, preventing an index
    error in the weight assignment.

    Args:
        data: List of record dicts, each containing a ``"label"`` key (0 or 1).
            Should be the training fold only.

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
    Builds the MONAI dictionary-based transform pipeline for loading and
    optionally augmenting 3D NIfTI volumes.

    MONAI uses dictionary-based transforms (transforms whose names end in ``d``,
    e.g. ``LoadImaged``) when a batch contains multiple keys. Here only the
    ``"image"`` key is transformed; the ``"label"`` key passes through unmodified.
    This keeps the pipeline extensible — segmentation masks or other metadata
    can be added to the same dict and transformed in sync with the image.

    **Baseline transforms (always applied):**

      1. ``LoadImaged``: reads the NIfTI file from disk into a NumPy array.
         MONAI automatically handles ``.nii`` and ``.nii.gz`` formats.

      2. ``EnsureChannelFirstd``: NIfTI volumes are stored as (H, W, D).
         PyTorch and MONAI expect (C, H, W, D). This transform inserts a
         channel dimension at position 0, producing a (1, H, W, D) tensor.

      3. ``Resized``: resizes the volume to ``spatial_size`` using trilinear
         interpolation. Note: since all volumes were already preprocessed to
         128³ by the data engine, this transform is effectively a no-op for
         the standard pipeline. It is retained so that the transform pipeline
         can be used with raw volumes of arbitrary size if needed.

      4. ``ScaleIntensityd``: linearly rescales voxel intensities to [0, 1]
         using the per-volume min and max. Raw CT and MRA intensities can span
         large ranges (e.g. -1000 to +3000 HU for CT) and vary across scanners.
         Normalising to [0, 1] ensures the model always receives inputs in a
         consistent range, which stabilises gradient magnitudes.

    **Augmentation transforms (training only, augment=True):**

    All augmentation transforms are stochastic — they fire with a given
    probability on each sample. They are applied only to the training set;
    validation and test sets see deterministic transforms only, so their
    metrics reflect true generalisation rather than a lucky random crop.

      5. ``RandFlipd`` (×3): randomly mirrors the volume along each spatial
         axis with 50% probability. Brain anatomy is approximately symmetric,
         so a left-right or anterior-posterior flip produces a plausible scan.
         Applying each axis independently yields up to 8 distinct orientations
         from a single volume.

      6. ``RandRotated``: randomly rotates the volume by up to ±0.26 radians
         (~15°) around each axis. Patients are not always positioned identically
         in the scanner; small rotations simulate this variability and discourage
         the model from relying on absolute orientation cues.

      7. ``RandScaleIntensityd``: multiplies all voxel intensities by a random
         factor in [1 - 0.1, 1 + 0.1] = [0.9, 1.1] with 50% probability.
         This simulates scanner-to-scanner gain variation and prevents the model
         from overfitting to the precise intensity values of training scans.
         Applied after ``ScaleIntensityd`` so the input is already in [0, 1].

      8. ``RandZoomd``: randomly scales the volume between 90% and 110% of its
         original size, then crops or pads to restore the original shape
         (``keep_size=True``). This simulates variability in subject distance
         from the detector and helps the model be invariant to small scale
         changes in apparent anatomy size.

      9. ``RandGaussianNoised``: adds zero-mean Gaussian noise with std=0.01
         with 20% probability. The noise is very mild (1% of the [0, 1] range)
         and acts as a regulariser, preventing the model from memorising exact
         voxel values.

    ``EnsureTyped`` at the end converts the result to a PyTorch tensor if it
    is not already one, ensuring a consistent output type regardless of which
    transforms were applied. When ``use_tabular=True``, the ``"tabular"`` key
    is also included in ``EnsureTyped`` so it is converted to a tensor.

    Args:
        spatial_size: Target (H, W, D) shape for ``Resized``. Defaults to
            ``(128, 128, 128)`` to match the data engine output.
        augment: If ``True``, appends the stochastic augmentation transforms.
            Should be ``True`` for training loaders and ``False`` for validation
            and test loaders.
        use_tabular: If ``True``, includes the ``"tabular"`` key in
            ``EnsureTyped`` so patient metadata is converted to a tensor.

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
            RandRotated(keys=keys, range_x=0.26, range_y=0.26, range_z=0.26, prob=0.5),
            RandScaleIntensityd(keys=keys, factors=0.1, prob=0.5),
            RandZoomd(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.3, keep_size=True),
            RandGaussianNoised(keys=keys, prob=0.2, mean=0.0, std=0.01),
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
    Partitions ``all_data`` into a stratified hold-out test set and a
    development set, then yields train/val/test triples.

    **Two-level split strategy:**

    The full dataset is first divided into a permanent hold-out *test set*
    and a *development set* (step 1). The test set is never used during
    training or hyperparameter selection — it exists solely to estimate
    generalisation performance after all training decisions have been made.

    The development set is then further divided into training and validation
    subsets (step 2). The validation set is used to monitor training progress,
    select the best model checkpoint (by F2-score), and inform hyperparameter
    choices. Because it influences training decisions, it cannot double as the
    test set.

    **Stratification:**

    Both splits use stratification, meaning the class ratio (positive :
    negative) is preserved in each subset. Without stratification, a small
    minority class could be concentrated in one split by chance, making metrics
    across folds incomparable and inflating variance.

    **k-fold vs single split:**

    When ``use_kfold=True`` (default), the development set is divided using
    stratified k-fold cross-validation. This yields ``n_splits`` folds, each
    using a different 1/k fraction as the validation set and the remaining
    (k-1)/k as training. Cross-validation gives a more reliable estimate of
    generalisation than a single split, at the cost of training k models.

    When ``use_kfold=False``, a single stratified train/val split is produced
    using ``val_split_ratio`` as the validation fraction. This is the mode to
    use during development and hyperparameter search.

    **Why a generator?**

    Returning a generator instead of a list means fold data is never all in
    memory at once. Each fold's data lists are only materialised when the caller
    advances the generator, which is important when the number of folds is large
    or the dataset dicts are expensive to hold.

    **Why no MONAI or PyTorch here?**

    This function works entirely with plain Python lists and scikit-learn. That
    makes it unit-testable on any machine without a GPU or a real dataset —
    synthetic lists of dicts are sufficient. MONAI and PyTorch dependencies are
    confined to ``build_dataloaders``.

    Args:
        all_data: Full list of record dicts as returned by ``get_data_list``.
        n_splits: Number of folds for k-fold cross-validation. Ignored when
            ``use_kfold=False``.
        test_size: Fraction of ``all_data`` to reserve as the hold-out test
            set (e.g. ``0.20`` for 20%).
        seed: Random seed for reproducible splits. Must match the seed used
            throughout the rest of the experiment.
        use_kfold: If ``True``, yield one tuple per k-fold split. If ``False``,
            yield a single tuple with a simple stratified train/val split.
        val_split_ratio: Fraction of the development set to use as validation
            when ``use_kfold=False``. Ignored when ``use_kfold=True``.

    Yields:
        Tuples of ``(fold_idx, train_files, val_files, test_files)`` where
        each ``*_files`` is a list of record dicts. ``test_files`` is identical
        across all folds — it is the same hold-out set throughout.
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
    use_tabular: bool = False,
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    cache_rate: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wraps train, val, and test record lists in MONAI Datasets and PyTorch
    DataLoaders, with optional oversampling for the training set.

    **Transform asymmetry:**

    The training loader uses ``get_transforms(augment=True)``; the validation
    and test loaders use ``get_transforms(augment=False)``. This asymmetry is
    intentional: augmentation is a regularisation technique that should only
    be applied during training. Evaluating on augmented inputs would measure
    performance on a different distribution than the real test data.

    **Oversampling with WeightedRandomSampler:**

    When ``oversample=True``, a ``WeightedRandomSampler`` is constructed that
    gives each sample a weight inversely proportional to its class frequency.
    PyTorch then draws ``len(train_files)`` samples with replacement from the
    training set each epoch, effectively up-sampling the minority class and
    down-sampling the majority class until the in-batch class distribution
    is balanced.

    ``replacement=True`` is required — without it, PyTorch would attempt to
    draw more samples than the minority class has, which is impossible without
    replacement.

    When a sampler is provided, ``shuffle`` must be ``False``: the two
    arguments are mutually exclusive in PyTorch (a sampler already determines
    the draw order; adding shuffle would create conflicting instructions and
    raise a ``ValueError``).

    **Worker count:**

    The training loader uses 4 workers and the evaluation loaders use 2.
    Training benefits from more parallel prefetching because each sample
    passes through heavier augmentation transforms. Validation and test
    transforms are lighter (no augmentation), so 2 workers are sufficient.

    **Reproducibility:**

    Each DataLoader receives an explicit ``generator`` seeded with ``seed``
    (controls the order in which batches are drawn from the main process) and
    a ``worker_init_fn`` (``seed_worker``) that re-seeds each worker's NumPy
    and Python RNGs so that augmentation is reproducible across runs.

    Args:
        train_files: Training fold record dicts.
        val_files: Validation fold record dicts.
        test_files: Hold-out test set record dicts.
        batch_size: Number of volumes per training batch.
        val_batch_size: Number of volumes per validation/test batch. Can be
            larger than ``batch_size`` because no gradients are computed.
        seed: Random seed for DataLoader generators and worker init.
        oversample: If ``True``, attach a ``WeightedRandomSampler`` to the
            training loader to balance the class distribution per epoch.
        use_tabular: If ``True``, the ``"tabular"`` key in each batch dict is
            converted to a tensor (requires records to already contain the
            ``"tabular"`` field from ``get_data_list``).
        spatial_size: Target ``(H, W, D)`` shape for the ``Resized`` transform.
            Defaults to ``(128, 128, 128)`` to preserve exact backwards
            compatibility with all existing experiments. Pass a different tuple
            to override the output resolution for all three DataLoaders.
        cache_rate: Fraction of the dataset to cache in RAM (0.0–1.0). When
            greater than zero, ``CacheDataset`` is used instead of ``Dataset``.
            MONAI's ``CacheDataset`` automatically detects the first random
            transform in the pipeline, caches all deterministic transforms
            (load, channel expansion, resize, intensity scaling), and applies
            the stochastic augmentations fresh at every ``__getitem__`` call.
            This eliminates repeated NIfTI I/O from epoch 2 onwards, which is
            the dominant runtime bottleneck for large volumes (256³). Set to
            ``0.0`` to revert to the original lazy-loading behaviour.

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
        train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=cache_rate, num_workers=4)
        val_ds = CacheDataset(data=val_files, transform=eval_transform, cache_rate=cache_rate, num_workers=2)
        test_ds = CacheDataset(data=test_files, transform=eval_transform, cache_rate=cache_rate, num_workers=2)
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

