## data_preprocess.py

`training_engine/src/data_preprocess.py` handles all data preparation for the training engine: NIfTI file discovery, stratified data splitting, class-imbalance weighting, MONAI transform construction, and DataLoader assembly.

**Transform library:** MONAI (`monai.transforms`, `monai.data`). No torchio, albumentations, or other augmentation libraries are used.

**Pipeline configurability:** The preprocessing pipeline is largely configurable from outside via function parameters (`spatial_size`, `augment`, `use_tabular`, `cache_rate`, `oversample`, `batch_size`, etc.). Augmentation probabilities, the specific transforms chosen, and their parameter ranges (e.g., zoom bounds, noise std) are hardcoded inside `get_transforms`. Resolution choices are constrained to a module-level allowlist (`VALID_RESOLUTIONS`).

---

### Module-level constant: `VALID_RESOLUTIONS`

A `Dict[str, Tuple[int, int, int]]` mapping human-readable `"HxWxD"` resolution strings (as they appear in config files) to the `(H, W, D)` integer tuples that MONAI's `Resized` transform expects. Currently holds four entries: `128x128x128`, `192x192x128`, `256x256x176`, `256x256x256`. Its purpose is to be a single source of truth — adding a new resolution requires only a new dict entry here, with no changes elsewhere in the module.

---

### Functions

---

#### `parse_input_resolution`

**What it does:** Validates a resolution config string and converts it to the `(H, W, D)` tuple required by MONAI's `Resized` transform.

**Design rationale:** Centralises resolution validation so that invalid strings fail early with a clear error message listing all valid options, rather than producing a cryptic shape mismatch downstream.

**Parameters:**
- `resolution` (`str`): A key from `VALID_RESOLUTIONS`, e.g. `"128x128x128"`.

**Returns:** `Tuple[int, int, int]` — spatial size `(H, W, D)`.

**Raises:** `ValueError` if `resolution` is not a key in `VALID_RESOLUTIONS`.

**Transforms / augmentations:** None.

---

#### `set_seed`

**What it does:** Seeds all random number generators (Python, NumPy, PyTorch CPU, PyTorch CUDA, MONAI) and enables cuDNN deterministic kernel selection for fully reproducible training runs.

**Design rationale:** Five independent RNG sources exist in a typical PyTorch + MONAI training loop; seeding all five is required to make data augmentation, weight initialisation, and batch ordering deterministic. `cudnn.deterministic = True` ensures the same conv algorithm is selected every run; `benchmark` is intentionally left enabled so auto-tuning can still optimise kernel selection within that constraint.

**Parameters:**
- `seed` (`int`, default `42`): Integer seed applied to all five RNG sources.

**Returns:** `None`.

**Transforms / augmentations:** None.

---

#### `seed_worker`

**What it does:** Seeds NumPy and Python's `random` module inside each DataLoader worker process, forwarding the per-worker seed that PyTorch derives from `torch.initial_seed()`.

**Design rationale:** PyTorch automatically seeds its own RNG per worker but does not propagate that seed to NumPy or `random`. Any MONAI transform that calls either library internally (e.g., random flip, Gaussian noise) will be non-deterministic across runs without this. Passed as `worker_init_fn` to every DataLoader constructed by `build_dataloaders`.

**Parameters:**
- `worker_id` (`int`): Worker process index, provided by PyTorch. Not used directly — `torch.initial_seed()` already incorporates it.

**Returns:** `None`.

**Transforms / augmentations:** None.

---

#### `get_data_list`

**What it does:** Walks a two-class directory layout (`root_dir/0/` and `root_dir/1/`) to discover all NIfTI files and returns them as a list of record dicts, optionally enriched with tabular features from a CSV.

**Design rationale:** Keeping file discovery as a pure data-gathering step (no transforms, no I/O beyond directory listing and optional CSV parsing) separates concerns cleanly. Using a flat `List[Dict]` as the record format is MONAI's idiomatic input for its dictionary-based transform API. Age is normalised by dividing by 100 (assumed patient age range) to bring it into `[0, 1]`, consistent with the image intensity range produced by `ScaleIntensityd`. Gender is encoded as `0.0 / 1.0`.

**Parameters:**
- `root_dir` (`str`): Path to the dataset root containing `0/` and `1/` subdirectories.
- `tabular_csv` (`Optional[str]`, default `None`): Path to a CSV with columns `exam`, `Age`, `gender` (matching `data_engine/dataset/classes.csv`). Rows with blank `Age` or `gender` are silently skipped.

**Returns:** `List[Dict]` — one record per NIfTI file. Each dict has:
- `"image"` (`str`): Absolute path to the NIfTI.
- `"label"` (`int`): Class label, `0` or `1`.
- `"path"` (`str`): Filename only.
- `"tabular"` (`np.ndarray`, shape `[2]`, dtype `float32`): `[age/100, gender]` — present only when `tabular_csv` is provided.

**Raises:** `KeyError` if a NIfTI filename (after stripping `.nii` / `.nii.gz`) has no matching row in the tabular CSV.

**Transforms / augmentations:** None (pure I/O and normalisation).

**Complexity flag:** The CSV parsing and NIfTI discovery are interleaved across two separate loops, connected only by the `tabular_csv is not None` guard repeated in both. If tabular support were extended (e.g., more features, different normalisation), the two sections would need to be updated together without any structural link enforcing that coupling.

---

#### `get_class_weights`

**What it does:** Computes inverse-frequency class weights (`total / count_c` per class) suitable for use as the `weight` argument to `torch.nn.CrossEntropyLoss`.

**Design rationale:** Inverse-frequency weighting causes gradient contributions from the minority class to be scaled up proportionally, counteracting the natural tendency of cross-entropy loss to minimise loss by predicting the majority class. This is one of two complementary imbalance strategies; the other is oversampling via `WeightedRandomSampler` in `build_dataloaders`. The function is intentionally scoped to the training fold only — computing weights on validation or test data would constitute label leakage.

**Parameters:**
- `data` (`List[Dict]`): Training fold record dicts, each containing a `"label"` key with value `0` or `1`.

**Returns:** `Optional[torch.Tensor]` — float tensor of shape `[2]` with `[weight_class_0, weight_class_1]`, or `None` if `data` is empty. Returns `0.0` weight for any class with zero samples (safe against `ZeroDivisionError`).

**Transforms / augmentations:** None.

---

#### `get_transforms`

**What it does:** Builds the full MONAI dictionary-based transform pipeline for 3D NIfTI volumes, toggling augmentation transforms on or off based on a flag.

**Design rationale:** A single function produces both training and evaluation pipelines by accepting an `augment` flag, ensuring the baseline transforms (load → channel → resize → scale → type-ensure) are identical between splits. All augmentations are spatial or intensity transforms commonly used in medical image analysis to expand the effective training set without requiring new acquisitions.

**Parameters:**
- `spatial_size` (`Tuple[int, int, int]`, default `(128, 128, 128)`): Target `(H, W, D)` for `Resized`.
- `augment` (`bool`, default `False`): If `True`, appends stochastic augmentation transforms.
- `use_tabular` (`bool`, default `False`): If `True`, includes `"tabular"` in the keys passed to `EnsureTyped`.

**Returns:** `monai.transforms.Compose` — the complete transform chain.

**Transforms and augmentations applied:**

| Transform | Always / Augment only | Purpose |
|---|---|---|
| `LoadImaged` | Always | Load NIfTI from disk into a NumPy array |
| `EnsureChannelFirstd` | Always | Add/confirm channel dimension: `(D,H,W)` → `(1,D,H,W)` |
| `Resized` | Always | Resize all volumes to `spatial_size` for uniform batch shapes |
| `ScaleIntensityd` | Always | Normalise intensity to `[0, 1]`; makes training stable across scanners |
| `RandFlipd` (×3 axes) | Augment only | Random left-right, anterior-posterior, superior-inferior flips; `prob=0.5` each; anatomically plausible for brain vasculature |
| `RandRotate90d` | Augment only | Discrete 0/90/180/270° axial rotation; `prob=0.5`, `max_k=3`; adds rotational invariance in the axial plane |
| `RandRotated` | Augment only | Continuous ±15° rotation in all three axes (`range_x/y/z=0.26` rad ≈ 15°); `prob=0.5` |
| `RandScaleIntensityd` | Augment only | Multiplicative intensity jitter by factor ±10%; `prob=0.5`; simulates scanner contrast variation |
| `RandZoomd` | Augment only | Random zoom 90–110% with `keep_size=True`; `prob=0.3`; simulates FOV and subject size variation |
| `RandGaussianNoised` | Augment only | Additive Gaussian noise `N(0, 0.05)` on `[0,1]`-scaled data; `prob=0.2`; simulates acquisition noise |
| `EnsureTyped` | Always (last) | Convert arrays to PyTorch tensors; applied to `["image"]` or `["image", "tabular"]` |

**Hardcoded augmentation parameters:** All probabilities, rotation ranges, zoom bounds, and noise std are fixed inside the function body. They are not exposed as parameters and cannot be changed from a config file without modifying this function.

---

#### `split_data`

**What it does:** Partitions the full dataset into stratified train/val/test subsets and yields one `(fold_idx, train_files, val_files, test_files)` tuple per fold (or a single tuple when k-fold is disabled).

**Design rationale:** A permanent held-out test set is carved out first (before any fold logic) so it is never seen during model selection — a standard ML practice. Stratification preserves the class imbalance ratio in every partition, preventing pathological splits where the minority class concentrates in one fold. The function is a generator so that the caller can iterate folds without materialising all splits at once.

**Parameters:**
- `all_data` (`List[Dict]`): Full record list from `get_data_list`.
- `n_splits` (`int`): Number of folds for k-fold CV; ignored when `use_kfold=False`.
- `test_size` (`float`): Fraction of `all_data` to hold out as the test set.
- `seed` (`int`, default `42`): Random seed for reproducible splits.
- `use_kfold` (`bool`, default `True`): If `True`, yields one tuple per `StratifiedKFold` fold; if `False`, yields one tuple from a single `train_test_split`.
- `val_split_ratio` (`float`, default `0.30`): Fraction of the development set to use as validation when `use_kfold=False`.

**Returns (yields):** `Generator[Tuple[int, List[Dict], List[Dict], List[Dict]], None, None]` — tuples of `(fold_idx, train_files, val_files, test_files)`.

**Transforms / augmentations:** None (pure index arithmetic).

**Complexity flag:** The stratification feasibility check (`can_stratify`) is computed twice — once for the full dataset and once for the dev subset (`can_stratify_dev`) — with no shared helper. The two checks use different patterns: the first counts via `labels.count(c)` inside a generator, the second repeats the same idiom conditionally. If stratification logic is ever extended (e.g., multi-class, minimum fold size checks), both sites must be updated. Extracting a small `_can_stratify(labels)` helper would eliminate this duplication.

---

#### `build_dataloaders`

**What it does:** Wraps train, val, and test record lists in MONAI `Dataset` or `CacheDataset` objects and returns three fully configured PyTorch `DataLoader`s.

**Design rationale:** Centralises all DataLoader configuration in one place so the training loop receives ready-to-use loaders without any setup logic. Training uses augmentation; val/test do not. `CacheDataset` (when `cache_rate > 0`) caches the output of deterministic transforms after epoch 1, eliminating repeated NIfTI disk I/O — particularly valuable on HPC clusters where storage I/O is the bottleneck. `WeightedRandomSampler` (when `oversample=True`) rebalances the training set at the batch-sampling level, complementing the loss-weighting approach in `get_class_weights`. `shuffle` is explicitly disabled when a sampler is attached because the two are mutually exclusive in PyTorch.

**Parameters:**
- `train_files` (`List[Dict]`): Training fold records.
- `val_files` (`List[Dict]`): Validation fold records.
- `test_files` (`List[Dict]`): Hold-out test set records.
- `batch_size` (`int`): Batch size for the training loader.
- `val_batch_size` (`int`): Batch size for val and test loaders.
- `seed` (`int`, default `42`): Seed for DataLoader generators and `seed_worker`.
- `oversample` (`bool`, default `False`): If `True`, attaches a `WeightedRandomSampler` to the training loader.
- `use_tabular` (`bool`, default `False`): Forwarded to `get_transforms` to include `"tabular"` in `EnsureTyped`.
- `spatial_size` (`Tuple[int, int, int]`, default `(128, 128, 128)`): Forwarded to `get_transforms` as the resize target.
- `cache_rate` (`float`, default `1.0`): Fraction of each dataset to cache in RAM. `0.0` disables caching entirely and uses plain `Dataset`.

**Returns:** `Tuple[DataLoader, DataLoader, DataLoader]` — `(train_loader, val_loader, test_loader)`.

**Transforms / augmentations:** Delegates entirely to `get_transforms`; training loader uses `augment=True`, val/test use `augment=False`.

**Hardcoded values:**
- Training `CacheDataset` uses `num_workers=4`; val/test use `num_workers=2` — fixed, not derived from system CPU count or a config parameter.
- Training `DataLoader` uses `num_workers=4`; val/test use `num_workers=2` — same hardcoding.
- `WeightedRandomSampler` always sets `num_samples=len(sample_weights)` (one full epoch worth of samples), which keeps epoch length equal to the training set size regardless of class imbalance.

**Complexity flag:** The `cache_rate > 0.0` branch duplicates the Dataset construction for all three splits (`train_ds`, `val_ds`, `test_ds`) in both the `if` and `else` arms — six near-identical lines. A small helper (e.g., `_make_dataset(files, transform, cache_rate)`) would collapse this to three calls and make it easier to add future Dataset variants (e.g., `PersistentDataset`).

---

### Complexity flags summary

| Location | Issue | Suggested simplification |
|---|---|---|
| `get_transforms` | All augmentation probabilities and parameter ranges are hardcoded inside the function body | Expose them as a dict/dataclass parameter (e.g., `aug_cfg`) so configs can tune them without code changes |
| `split_data` | Stratification feasibility check duplicated for full dataset and dev subset | Extract a `_can_stratify(labels)` helper used in both places |
| `build_dataloaders` | Dataset construction repeated in both `cache_rate > 0` and `else` branches (6 near-identical lines) | Extract a `_make_dataset(files, transform, cache_rate)` helper |
| `build_dataloaders` | `num_workers` values (4 for train, 2 for val/test) hardcoded in two places (CacheDataset and DataLoader) | Derive from a parameter or `os.cpu_count()`, or at minimum define as module-level constants |
| `get_data_list` | CSV parsing and NIfTI discovery are two separate loops with repeated `tabular_csv is not None` guards | Separate into `_load_tabular_csv(path)` and `_discover_niftis(root_dir)`, then compose in `get_data_list` |

---

