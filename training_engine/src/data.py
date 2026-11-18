"""
Data loading, splitting, transforms.

Datasets come pre-resampled from data_engine (iso_128 at 128^3, iso_256 at
256^3), so no Resize transform is needed here — just load, normalise, augment.

Seeding, k-fold splitting, and class weighting are generic and live in
``dlorchestra`` — re-exported here for backward compatibility with existing
callers (``set_seed``, ``split_data(stratify_by_scanner=...)``,
``get_class_weights``).
"""

import csv
import logging
import os
import time
import zlib
from typing import Generator

import numpy as np
import torch
from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandCoarseDropoutd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
)

from dlorchestra.seeding import seed_worker as _seed_worker, set_seed
from dlorchestra.splitting import get_class_weights
from dlorchestra.splitting import split_data as _dl_split_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _load_tabular_csv(csv_path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Load tabular metadata CSV into a dict keyed by exam ID.

    Returns {exam_id: np.array([age/100, gender], dtype=float32)}.
    Age is divided by 100 to bring it into roughly [0, 1].
    Rows with missing Age or gender are skipped.
    """
    lookup: dict[str, np.ndarray] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            age_str = row.get("Age", "").strip()
            gender_str = row.get("gender", "").strip()
            if not age_str or not gender_str:
                continue
            exam_id = row["exam"].strip()
            lookup[exam_id] = np.array(
                [float(age_str) / 100.0, float(gender_str)],
                dtype=np.float32,
            )
    return lookup


_SUPPORTED_EXTENSIONS = (".nii", ".nii.gz", ".npy")


def _strip_extension(fname: str) -> str:
    """Strip supported file extensions from filename."""
    for ext in (".nii.gz", ".nii", ".npy"):
        if fname.endswith(ext):
            return fname[: -len(ext)]
    return fname


def _strip_series_suffix(exam_id: str) -> str:
    """Strip the multi-series letter suffix from an exam ID (e.g. BP0024A -> BP0024)."""
    return exam_id.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or exam_id


def get_data_list(
    root_dir: str | os.PathLike,
    use_multimodal: bool = False,
) -> list[dict]:
    """Walk root_dir/0/ and root_dir/1/ to build a MONAI-style record list.

    Returns a list of dicts with keys:
      - "image": absolute path (NIfTI or .npy, consumed by transforms)
      - "label": int (0 or 1)
      - "path": absolute path (preserved through transforms for predictions.csv)
      - "tabular": np.ndarray [age/100, gender] (only when use_multimodal=True)
    """
    if use_multimodal:
        from config import DATASET_CSV
        tabular_lookup = _load_tabular_csv(DATASET_CSV)
    else:
        tabular_lookup = None

    root = str(root_dir)
    data_list: list[dict] = []
    for class_label in (0, 1):
        class_dir = os.path.join(root, str(class_label))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(_SUPPORTED_EXTENSIONS):
                abs_path = os.path.join(class_dir, fname)
                record: dict = {
                    "image": abs_path,
                    "label": class_label,
                    "path": abs_path,
                }
                if tabular_lookup is not None:
                    exam_id = _strip_extension(fname)
                    base_id = _strip_series_suffix(exam_id)
                    lookup_id = exam_id if exam_id in tabular_lookup else base_id
                    if lookup_id not in tabular_lookup:
                        raise KeyError(
                            f"No tabular data for exam '{exam_id}' "
                            f"(file: {fname})"
                        )
                    record["tabular"] = tabular_lookup[lookup_id]
                data_list.append(record)
    return data_list


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _load_scanner_lookup(csv_path: str | os.PathLike) -> dict[str, str]:
    """Load {exam_id: "brand_fieldT"} scanner group from the metadata CSV."""
    lookup: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            exam_id = row["exam"].strip()
            brand = row.get("brand", "").strip()
            field = row.get("field_strength_T", "").strip()
            lookup[exam_id] = f"{brand}_{field}T" if brand and field else "unknown"
    return lookup


def _exam_id(record: dict) -> str:
    """Extract exam ID from record path, stripping multi-series suffix."""
    fname = os.path.basename(record["path"])
    exam_id = _strip_extension(fname)
    return _strip_series_suffix(exam_id)


def split_data(
    all_data: list[dict],
    n_splits: int = 5,
    test_size: float = 0.15,
    seed: int = 42,
    use_kfold: bool = True,
    val_split_ratio: float = 0.20,
    stratify_by_scanner: bool = False,
) -> Generator[tuple[int, list[dict], list[dict], list[dict]], None, None]:
    """Stratified split with a permanent held-out test set.

    When stratify_by_scanner is True, stratification uses a composite
    "{label}_{brand}_{field_strength}" key (read from DATASET_CSV) instead
    of just the class label, so each fold gets proportional representation
    of scanner groups. Requires records to carry a "path" key so the exam
    id can be looked up in the CSV.

    Yields (fold_idx, train_files, val_files, test_files) tuples.
    The test set is identical across all folds.

    Thin wrapper around dlorchestra.splitting.split_data — supplies the
    scanner-lookup group_fn this project uses for stratify_by_scanner.
    """
    group_fn = None
    if stratify_by_scanner:
        from config import DATASET_CSV
        scanner_lookup = _load_scanner_lookup(DATASET_CSV)
        group_fn = lambda rec: scanner_lookup.get(_exam_id(rec), "unknown")  # noqa: E731

    return _dl_split_data(
        all_data,
        n_splits=n_splits,
        test_size=test_size,
        seed=seed,
        use_kfold=use_kfold,
        val_split_ratio=val_split_ratio,
        group_fn=group_fn,
    )


# ---------------------------------------------------------------------------
# Normalization transforms
# ---------------------------------------------------------------------------

SPHERE_RADIUS = 2.5  # ~5 voxel diameter; shared with scripts/add_synthetic_sphere.py


def synthetic_sphere_seed(seed: int, path: str) -> int:
    """Deterministic per-file rng seed from a file path.

    Uses crc32 (not the built-in hash()) because str hashing is salted
    per-process — not reproducible across runs/workers.
    """
    return seed + zlib.crc32(path.encode())


def sample_synthetic_sphere_mask(arr: np.ndarray, rng: np.random.Generator, radius: float) -> np.ndarray:
    """Boolean mask for a bright sphere of the given radius, centered at a
    random point within the central 50% of arr's nonzero (brain) bounding
    box — or arr's geometric center if there's no nonzero signal.
    """
    brain = np.argwhere(arr > 0)
    if brain.size == 0:
        center = np.array(arr.shape) // 2
    else:
        lo, hi = brain.min(axis=0), brain.max(axis=0)
        margin = (hi - lo) * 0.25
        center = rng.uniform(lo + margin, hi - margin)

    zz, yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return dist2 <= radius ** 2


class AddSyntheticSphered(MapTransform):
    """Stamp a bright sphere onto class-1 volumes (sanity check transform).

    Only modifies volumes where record["label"] == 1. The sphere has a
    random position within the central 50% of the brain bounding box and
    intensity equal to the volume's max value.
    """

    def __init__(self, keys, radius: float = SPHERE_RADIUS, seed: int = 42):
        super().__init__(keys)
        self.radius = radius
        self.seed = seed

    def __call__(self, data):
        d = dict(data)
        if d.get("label") != 1:
            return d
        for key in self.keys:
            vol = d[key]
            # vol shape is (C, D, H, W) after EnsureChannelFirst
            arr = vol[0] if isinstance(vol, (np.ndarray, torch.Tensor)) else vol
            if isinstance(arr, torch.Tensor):
                arr_np = arr.numpy()
            else:
                arr_np = arr

            rng = np.random.default_rng(synthetic_sphere_seed(self.seed, d.get("path", "")))
            sphere_mask = sample_synthetic_sphere_mask(arr_np, rng, self.radius)

            if isinstance(vol, torch.Tensor):
                vol[0][torch.from_numpy(sphere_mask)] = float(arr_np.max())
            else:
                vol[0][sphere_mask] = arr_np.max()
            d[key] = vol
        return d


class LoadNpyd(MapTransform):
    """Load .npy files and ensure channel-first layout.

    2D arrays (H, W) become (1, H, W). 3D arrays (C, H, W) are kept as-is,
    or reduced to `channels` (e.g. [0] = axial view of a 3ch MIP).
    """

    def __init__(self, keys, channels: list[int] | None = None):
        super().__init__(keys)
        self.channels = channels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = np.load(d[key]).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]  # (H, W) → (1, H, W)
            if self.channels is not None:
                arr = arr[self.channels]
            d[key] = arr
        return d


class PercentileNormalized(MapTransform):
    """Clip to [p1, p99] of foreground, then scale to [0, 1]."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vol = d[key]
            fg = vol[vol > 0]
            if len(fg) == 0:
                continue
            p1, p99 = np.percentile(fg, [1, 99])
            vol = np.clip(vol, 0, p99)
            vol = np.where(vol > 0, (vol - p1) / max(p99 - p1, 1e-9), 0.0)
            d[key] = np.clip(vol, 0, None).astype(np.float32)
        return d


class ZScoreNormalized(MapTransform):
    """Z-score normalization of foreground voxels. Background stays zero."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vol = d[key]
            fg = vol[vol > 0]
            if len(fg) == 0:
                continue
            mu, sigma = fg.mean(), max(fg.std(), 1e-9)
            d[key] = np.where(vol > 0, (vol - mu) / sigma, 0.0).astype(np.float32)
        return d


class HistogramMatched(MapTransform):
    """Match foreground intensity histogram to a precomputed reference template."""

    def __init__(self, keys, template_path):
        super().__init__(keys)
        from skimage.exposure import match_histograms
        self._match_histograms = match_histograms
        self._template = np.load(template_path).astype(np.float32)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vol = d[key]
            mask = vol > 0
            fg = vol[mask]
            if len(fg) == 0:
                continue
            matched_fg = self._match_histograms(fg, self._template, channel_axis=None)
            out = np.zeros_like(vol)
            out[mask] = matched_fg.astype(np.float32)
            d[key] = out
        return d


class RandRicianNoised(MapTransform):
    """Add Rician noise to MRI magnitude images.

    Rician noise: M = sqrt((S + n1)² + n2²) where n1, n2 ~ N(0, σ²).
    At high SNR it approximates Gaussian; at low SNR (near background) it
    has a positive bias that Gaussian doesn't model.
    """

    def __init__(self, keys, std: float = 0.05, prob: float = 0.2):
        super().__init__(keys)
        self.std = std
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if np.random.rand() > self.prob:
            return d
        for key in self.keys:
            vol = d[key]
            if isinstance(vol, torch.Tensor):
                n_real = torch.randn_like(vol) * self.std
                n_imag = torch.randn_like(vol) * self.std
                d[key] = torch.sqrt((vol + n_real) ** 2 + n_imag ** 2)
            else:
                n_real = np.random.normal(0, self.std, size=vol.shape).astype(vol.dtype)
                n_imag = np.random.normal(0, self.std, size=vol.shape).astype(vol.dtype)
                d[key] = np.sqrt((vol + n_real) ** 2 + n_imag ** 2)
        return d


NORMALIZATION_TRANSFORMS = {
    "percentile": PercentileNormalized,
    "zscore": ZScoreNormalized,
    "histogram": HistogramMatched,
    "none": None,
}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(
    config: dict, augment: bool = False, use_tabular: bool = False,
) -> Compose:
    """Build the MONAI transform pipeline.

    Supports both 3D volumes (NIfTI) and 2D images (.npy) via SPATIAL_DIMS.
    All augmentation parameters are read from config so they can be tuned
    per-experiment without code changes.
    """
    norm_key = config.get("normalization", "percentile")
    norm_cls = NORMALIZATION_TRANSFORMS[norm_key]
    spatial_dims = config.get("SPATIAL_DIMS", 3)

    transforms = []

    # -- loading --
    if spatial_dims == 2:
        transforms.append(LoadNpyd(keys=["image"], channels=config.get("INPUT_CHANNELS")))
    else:
        transforms.append(LoadImaged(keys=["image"]))
        transforms.append(EnsureChannelFirstd(keys=["image"]))

    # -- synthetic sphere (3D only) --
    if spatial_dims == 3 and config.get("ADD_SYNTHETIC_SPHERE", False):
        transforms.append(AddSyntheticSphered(
            keys=["image"],
            radius=config.get("SYNTHETIC_SPHERE_RADIUS", 2.5),
            seed=config.get("RANDOM_SEED", 42),
        ))

    # -- optional resize (e.g. ViT needs 224x224) --
    resize_to = config.get("RESIZE_TO")
    if resize_to and spatial_dims == 2:
        from monai.transforms import Resized
        transforms.append(Resized(
            keys=["image"],
            spatial_size=resize_to,
            mode="bilinear",
        ))

    # -- normalization (dimension-agnostic) --
    if norm_cls is not None:
        if norm_key == "histogram":
            transforms.append(norm_cls(keys=["image"], template_path=config["histogram_template"]))
        else:
            transforms.append(norm_cls(keys=["image"]))

    # -- augmentation --
    if augment:
        # Flips: one per spatial axis (2 for 2D, 3 for 3D)
        for axis in range(spatial_dims):
            transforms.append(
                RandFlipd(keys=["image"], spatial_axis=axis, prob=config["AUG_FLIP_PROB"])
            )
        transforms.extend([
            RandRotate90d(keys=["image"], prob=config["AUG_ROT90_PROB"], max_k=3),
            RandScaleIntensityd(
                keys=["image"],
                factors=config["AUG_SCALE_INTENSITY_RANGE"],
                prob=config["AUG_SCALE_INTENSITY_PROB"],
            ),
            RandZoomd(
                keys=["image"],
                min_zoom=config["AUG_ZOOM_MIN"],
                max_zoom=config["AUG_ZOOM_MAX"],
                prob=config["AUG_ZOOM_PROB"],
                keep_size=True,
            ),
            RandGaussianNoised(
                keys=["image"],
                std=config["AUG_NOISE_STD"],
                prob=config["AUG_NOISE_PROB"],
            ),
        ])
        # Continuous rotation (2D only — safe for MIP, avoids 3D interp artifacts)
        if spatial_dims == 2:
            rotate_prob = config.get("AUG_ROTATE_PROB", 0.0)
            if rotate_prob > 0:
                rot_range = config.get("AUG_ROTATE_RANGE", 0.2618)
                transforms.append(RandRotated(
                    keys=["image"],
                    range_x=rot_range,
                    prob=rotate_prob,
                    keep_size=True,
                    padding_mode="zeros",
                ))
        # Random cutout / coarse dropout — disabled by default (prob=0.0)
        cutout_prob = config.get("AUG_CUTOUT_PROB", 0.0)
        if cutout_prob > 0:
            hole_size = config.get("AUG_CUTOUT_SIZE", 40)
            spatial_size = [hole_size] * spatial_dims
            transforms.append(RandCoarseDropoutd(
                keys=["image"],
                holes=config.get("AUG_CUTOUT_HOLES", 3),
                spatial_size=spatial_size,
                fill_value=0.0,
                prob=cutout_prob,
            ))
        # Gaussian blur / smoothing — disabled by default (prob=0.0)
        blur_prob = config.get("AUG_BLUR_PROB", 0.0)
        if blur_prob > 0:
            sigma_min = config.get("AUG_BLUR_SIGMA_MIN", 0.5)
            sigma_max = config.get("AUG_BLUR_SIGMA_MAX", 1.5)
            transforms.append(RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(sigma_min, sigma_max),
                sigma_y=(sigma_min, sigma_max),
                prob=blur_prob,
            ))
        # Rician noise (MRI-realistic, 3D only) — disabled by default (prob=0.0)
        if spatial_dims == 3:
            rician_prob = config.get("AUG_RICIAN_NOISE_PROB", 0.0)
            if rician_prob > 0:
                transforms.append(RandRicianNoised(
                    keys=["image"],
                    std=config["AUG_RICIAN_NOISE_STD"],
                    prob=rician_prob,
                ))

    ensure_keys = ["image", "tabular"] if use_tabular else ["image"]
    transforms.append(EnsureTyped(keys=ensure_keys))
    return Compose(transforms)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_files: list[dict],
    val_files: list[dict],
    test_files: list[dict],
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders.

    Training uses augmentation and optional oversampling; val/test do not.
    CacheDataset caches deterministic transforms after epoch 1.
    """
    use_tabular = "tabular" in train_files[0] if train_files else False
    train_transform = get_transforms(config, augment=True, use_tabular=use_tabular)
    eval_transform = get_transforms(config, augment=False, use_tabular=use_tabular)

    cache_rate = config["CACHE_RATE"]
    num_workers = config["NUM_WORKERS"]
    seed = config["RANDOM_SEED"]

    def _make_dataset(files, transform, label="dataset"):
        t0 = time.time()
        if cache_rate > 0:
            ds = CacheDataset(
                data=files,
                transform=transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
                progress=False,
            )
        else:
            ds = Dataset(data=files, transform=transform)
        logger.info("Cached %s (%d samples) in %.1fs", label, len(files), time.time() - t0)
        return ds

    train_ds = _make_dataset(train_files, train_transform, "train")
    val_ds = _make_dataset(val_files, eval_transform, "val")
    test_ds = _make_dataset(test_files, eval_transform, "test")

    # -- oversampling (WeightedRandomSampler) --
    sampler = None
    shuffle_train = True
    if config["balancing"] == "oversampling":
        labels = [d["label"] for d in train_files]
        counts = np.bincount(labels, minlength=2).astype(float)
        sample_weights = [1.0 / counts[l] for l in labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle_train = False  # mutually exclusive with sampler

    # Python 3.14 defaults to forkserver, which sends each cached tensor as an fd and
    # fails past ~256 ("too many fds"); fork shares the cache copy-on-write instead
    mp_ctx = {"multiprocessing_context": "fork"} if num_workers > 0 else {}
    # seeded generator per loader for reproducible batch ordering
    train_loader = DataLoader(
        train_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=shuffle_train,
        num_workers=num_workers,
        sampler=sampler,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **mp_ctx,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **mp_ctx,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **mp_ctx,
    )
    return train_loader, val_loader, test_loader
