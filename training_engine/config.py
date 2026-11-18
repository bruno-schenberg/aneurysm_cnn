import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAINING_ENGINE_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = TRAINING_ENGINE_ROOT / "experiments"
CONFIGS_DIR = TRAINING_ENGINE_ROOT / "configs"

# Metadata CSV — shared with data_engine (exam, class, Age, gender, ...).
# Override with env var on the cluster.
DATASET_CSV = Path(os.environ.get(
    "DATASET_CSV",
    str(Path(__file__).parent.parent / "dataset" / "dataset.csv"),
))

def _dataset_path(env_var: str, default_name: str) -> Path:
    """Dataset dir: env var override, else $DATASETS_ROOT/<default_name> (default /mnt/data/datasets)."""
    root = os.environ.get("DATASETS_ROOT", "/mnt/data/datasets")
    return Path(os.environ.get(env_var, f"{root}/{default_name}"))


# Dataset paths — override with env vars on the cluster.
DATASET_PATHS: dict[str, Path] = {
    "iso_128": _dataset_path("DATASET_ISO_128", "iso_128"),
    "iso_256": _dataset_path("DATASET_ISO_256", "iso_256"),
    "sample_iso_128": _dataset_path("DATASET_SAMPLE_ISO_128", "sample_iso_128"),
    "modelnet10_pair": _dataset_path("DATASET_MODELNET10_PAIR", "modelnet10_pair"),
    "aniso_384": _dataset_path("DATASET_ANISO_384", "aniso_384"),
    "aniso_512": _dataset_path("DATASET_ANISO_512", "aniso_512"),
    "c90z_256": _dataset_path("DATASET_C90Z_256", "c90z_256"),
    "vessel90z_256": _dataset_path("DATASET_VESSEL90Z_256", "vessel90z_256"),
    "c90z_384": _dataset_path("DATASET_C90Z_384", "c90z_384"),
    # Phase 7A cascade stage 2 — patches cropped around nnU-Net candidate
    # centroids (segmentation/build_patch_dataset.py), not a data_engine variant.
    "patch_dataset_64": _dataset_path("DATASET_PATCH_64", "patch_dataset_64"),
    "mip_axial": _dataset_path("DATASET_MIP_AXIAL", "mip_axial"),
    "mip_coronal": _dataset_path("DATASET_MIP_CORONAL", "mip_coronal"),
    "mip_sagittal": _dataset_path("DATASET_MIP_SAGITTAL", "mip_sagittal"),
    "mip_3ch": _dataset_path("DATASET_MIP_3CH", "mip_3ch"),
    # Mixed datasets (our data + Lausanne)
    "mip_axial_mixed": _dataset_path("DATASET_MIP_AXIAL_MIXED", "mip_axial_mixed"),
    "mip_3ch_mixed": _dataset_path("DATASET_MIP_3CH_MIXED", "mip_3ch_mixed"),
    "mip_sagittal_mixed": _dataset_path("DATASET_MIP_SAGITTAL_MIXED", "mip_sagittal_mixed"),
    # Frangi-enhanced MIPs (3D Frangi on volume, then MIP)
    "mip_frangi_2ch": _dataset_path("DATASET_MIP_FRANGI_2CH", "mip_frangi_2ch"),
    "mip_frangi_diff": _dataset_path("DATASET_MIP_FRANGI_DIFF", "mip_frangi_diff"),
    # Frangi-enhanced MIPs (2D Frangi on MIP image)
    "mip_frangi2d_2ch": _dataset_path("DATASET_MIP_FRANGI2D_2CH", "mip_frangi2d_2ch"),
    "mip_frangi2d_diff": _dataset_path("DATASET_MIP_FRANGI2D_DIFF", "mip_frangi2d_diff"),
    # Top-Z-cropped MIPs (from the 180 mm centroid-cropped NIfTIs at native spacing, not c90z;
    # top N% of slices, so the physical window varies per exam)
    "mip_3ch_top90": _dataset_path("DATASET_MIP_3CH_TOP90", "mip_3ch_top90"),
    "mip_3ch_top80": _dataset_path("DATASET_MIP_3CH_TOP80", "mip_3ch_top80"),
    "mip_3ch_top70": _dataset_path("DATASET_MIP_3CH_TOP70", "mip_3ch_top70"),
    "mip_3ch_top60": _dataset_path("DATASET_MIP_3CH_TOP60", "mip_3ch_top60"),
    # Top-Z-cropped mixed (our + Lausanne)
    "mip_3ch_top90_mixed": _dataset_path("DATASET_MIP_3CH_TOP90_MIXED", "mip_3ch_top90_mixed"),
    # Phase 6: higher-resolution MIPs
    "mip_3ch_top90_512": _dataset_path("DATASET_MIP_3CH_TOP90_512", "mip_3ch_top90_512"),
    "mip_3ch_top90_768": _dataset_path("DATASET_MIP_3CH_TOP90_768", "mip_3ch_top90_768"),
    # Phase 6: thin-slab MIPs (axial, N slabs as channels)
    "mip_thinslab5": _dataset_path("DATASET_MIP_THINSLAB5", "mip_thinslab5"),
    # Phase 6: multi-angle MIPs (rotated projections as channels)
    "mip_multiangle6": _dataset_path("DATASET_MIP_MULTIANGLE6", "mip_multiangle6"),
    # Phase 7C: vessel-aware 3D crop
    "vessel_crop_256": _dataset_path("DATASET_VESSEL_CROP_256", "vessel_crop_256"),
}

# Final-campaign MIP grid (data_engine/steps/step_mip/generate_mip_grid.py):
# key "<family>/<window>_<aspect>_<size>" -> $DATASETS_ROOT/<family>/<window>_<aspect>_<size>.
# Families: ours, Lausanne-only, ours+Lausanne (symlink view), and the 2D-1b
# preprocessing variants. Windows must match generate_mip_grid.WINDOWS.
MIP_GRID_FAMILIES = ("mip_grid", "mip_grid_lausanne", "mip_grid_mixed",
                     "mip_grid_n4", "mip_grid_n4mask")
MIP_GRID_WINDOWS = ("center90", "top60", "top75", "top90mm", "top120",
                    "vessel60", "vessel90", "pct90", "full")
DATASET_PATHS.update({
    f"{fam}/{w}_{a}_{s}": _dataset_path("_unused", f"{fam}/{w}_{a}_{s}")
    for fam in MIP_GRID_FAMILIES for w in MIP_GRID_WINDOWS
    for a in ("stretch", "pad") for s in (384, 512, 768)
})

# Expected spatial resolution per dataset key, used as a sanity check.
# Must match data_engine VARIANTS config (spacing = 180mm / dim). Only
# covers the 3D-volume dataset keys above (10 of 28) — the 2D/MIP keys
# (mip_*) aren't volumes and don't have a single "spatial resolution" this
# check applies to (their shape is channel-count x MIP target-size instead).
DATASET_RESOLUTIONS: dict[str, tuple[int, int, int]] = {
    "iso_128": (128, 128, 128),
    "iso_256": (256, 256, 256),
    "sample_iso_128": (128, 128, 128),
    "modelnet10_pair": (128, 128, 128),
    "aniso_384": (384, 384, 128),
    "aniso_512": (512, 512, 128),
    "c90z_256": (256, 256, 128),
    "vessel90z_256": (256, 256, 128),
    "c90z_384": (384, 384, 192),
    "patch_dataset_64": (64, 64, 64),
    "vessel_crop_256": (256, 256, 128),
}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = "cuda" if os.environ.get("FORCE_CPU") is None else "cpu"

# ---------------------------------------------------------------------------
# Default experiment config
# ---------------------------------------------------------------------------
# Every experiment JSON entry is merged over this dict. Keys here are the
# complete set the codebase reads — no undocumented extras.

DEFAULT_CONFIG: dict = {
    # -- identity (required per-experiment, no useful default) --
    "name": None,
    "model": None,
    "data_path_key": None,

    # -- training --
    "EPOCHS": 100,
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "GRAD_ACCUM_STEPS": 1,
    "USE_AMP": True,
    # "bfloat16" (default, no scaler needed) or "float16" (uses GradScaler).
    # On ROCm, bfloat16 triggers MIOpen kernel bugs at 256³ — use float16.
    "AMP_DTYPE": "bfloat16",
    # Max seconds per epoch. 0 = no limit. Aborts experiment if exceeded.
    "EPOCH_TIMEOUT": 0,
    # torch.compile the model before training. Requires triton on ROCm.
    "TORCH_COMPILE": False,
    "EARLY_STOPPING_PATIENCE": 15,
    # "val_f2" (recall-weighted, better for clinical imbalance) or "val_auc"
    "CHECKPOINT_METRIC": "val_f2",

    # -- lr scheduler --
    # "cosine" (CosineAnnealingWarmRestarts) or "none"
    "LR_SCHEDULER": "cosine",
    "COSINE_T0": 10,           # restart period in epochs
    "COSINE_T_MULT": 2,        # period multiplier after each restart
    "COSINE_ETA_MIN": 1e-6,    # minimum lr
    # Linear LR ramp from 1% to 100% over this many epochs before the schedule
    # above starts (0 = off). Transformers (ViT/Swin) need it on small data.
    "WARMUP_EPOCHS": 0,

    # -- data split --
    "USE_KFOLD": False,
    "N_SPLITS": 5,
    "TEST_SPLIT_RATIO": 0.15,
    "VAL_SPLIT_RATIO": 0.20,
    "STRATIFY_BY_SCANNER": False,

    # -- class imbalance: "weighted_cost_function" | "oversampling" | "none" --
    "balancing": "weighted_cost_function",

    # -- loss function: "cross_entropy" | "focal" --
    "LOSS_FN": "cross_entropy",
    "FOCAL_GAMMA": 2.0,

    # -- dataloader --
    "NUM_WORKERS": 4,
    "CACHE_RATE": 1.0,

    # -- normalization: "percentile" | "zscore" | "histogram" | "none" --
    "normalization": "percentile",
    # Path to .npy template file (required when normalization=histogram).
    "histogram_template": str(
        Path(__file__).parent.parent / "dataset" / "histogram_template.npy"
    ),

    # -- augmentation (all probs and ranges exposed) --
    "AUG_FLIP_PROB": 0.5,
    "AUG_ROT90_PROB": 0.5,
    "AUG_SCALE_INTENSITY_PROB": 0.5,
    "AUG_SCALE_INTENSITY_RANGE": 0.1,
    "AUG_ZOOM_PROB": 0.3,
    "AUG_ZOOM_MIN": 0.9,
    "AUG_ZOOM_MAX": 1.1,
    "AUG_NOISE_PROB": 0.2,
    "AUG_NOISE_STD": 0.05,
    # Rician noise (MRI-realistic). Prob=0.0 means disabled by default.
    # When enabled, applied IN ADDITION to Gaussian noise.
    "AUG_RICIAN_NOISE_PROB": 0.0,
    "AUG_RICIAN_NOISE_STD": 0.05,
    # Continuous rotation (2D only — safe for MIP, avoids 3D interp artifacts)
    "AUG_ROTATE_PROB": 0.0,
    "AUG_ROTATE_RANGE": 0.2618,  # ~15 degrees in radians
    # Random cutout / coarse dropout (rectangular holes)
    "AUG_CUTOUT_PROB": 0.0,
    "AUG_CUTOUT_HOLES": 3,
    "AUG_CUTOUT_SIZE": 40,  # max hole size in pixels
    # Gaussian blur / smoothing
    "AUG_BLUR_PROB": 0.0,
    "AUG_BLUR_SIGMA_MIN": 0.5,
    "AUG_BLUR_SIGMA_MAX": 1.5,

    # -- classifier head dropout (2D ResNet models) --
    "DROPOUT": 0.5,

    # -- backbone freezing --
    # Freeze backbone for the first N epochs, only training the classifier
    # head. 0 = disabled (all params trainable from epoch 1).
    "FREEZE_BACKBONE_EPOCHS": 0,

    # -- reproducibility --
    "RANDOM_SEED": 42,
    # Seed for the train/val/test split; None = RANDOM_SEED. Set RANDOM_SEED
    # alone to repeat a run with new init/augmentation on the same folds.
    "SPLIT_SEED": None,
    # cudnn.benchmark auto-tunes convolution kernels. Disable on ROCm if
    # MIOpen kernel compilation fails (e.g. BatchNorm backward on 256³).
    "CUDNN_BENCHMARK": True,

    # -- multimodal fusion --
    # When True, loads age/gender from DATASET_CSV and attaches to each
    # record for late fusion with a tabular branch.
    "USE_MULTIMODAL": False,

    # -- evaluation --
    # Test set is always split out (indices locked by seed, TEST_SPLIT_RATIO).
    # EVAL_ON_TEST controls whether it's loaded and evaluated on.
    # False (default) = eval on val only — use during experimentation.
    # True = load + eval held-out test set — use for final runs.
    "EVAL_ON_TEST": False,
    # Frozen test set (repo-relative JSON list of exam IDs, e.g.
    # "dataset/test_ids.json"). When set, the run aborts before training if
    # its test split differs. Final runs set it; None = no check.
    "TEST_IDS_FILE": None,
    # dataset/exclusions.json: drop its "exclude" exams, frozen TEST_IDS_FILE test set,
    # patient-grouped folds (src/split.py). None = dlorchestra's default split.
    "EXCLUSIONS_FILE": None,

    # -- synthetic sphere sanity check --
    # Stamp a bright sphere on class-1 volumes at runtime (no extra files).
    "ADD_SYNTHETIC_SPHERE": False,
    "SYNTHETIC_SPHERE_RADIUS": 2.5,  # ~5 voxel diameter

    # -- spatial dimensionality --
    # 3 = standard 3D volume pipeline (NIfTI + 3D augments + 3D models)
    # 2 = 2D image pipeline (numpy .npy + 2D augments + 2D models, e.g. MIP)
    "SPATIAL_DIMS": 3,

    # -- input channels (2D models only) --
    # Number of channels for models that accept variable input (RESNET50_2D_NCH).
    # Ignored by fixed-channel models (e.g. RESNET50_2D_3CH always uses 3).
    "IN_CHANNELS": 3,
    # Channels of the .npy to keep (2D only), e.g. [0] = axial of a 3ch MIP for
    # RESNET50_2D. None = all.
    "INPUT_CHANNELS": None,

    # -- optional image resize (2D only) --
    # [H, W] to resize after loading. Use for ViT (needs 224x224).
    # None = no resize (use native MIP resolution).
    "RESIZE_TO": None,

    # -- MIL (Multiple Instance Learning) --
    # Patch size for MIL models. Volume is split into non-overlapping
    # P^3 patches. 128^3 with P=64 → 8 patches; 256^3 with P=64 → 64 patches.
    "MIL_PATCH_SIZE": 64,

    # -- external data --
    # When set, files whose basename starts with this prefix are treated as
    # external data: included in training only, never in val/test.
    # Use with mixed datasets (e.g. "laus_" for Lausanne data).
    "EXTERNAL_DATA_PREFIX": None,

    # -- misc --
    # len(CLASSES) becomes num_classes (dlorchestra.experiment.TorchExperiment
    # passes it to build_model/get_class_weights). Changing this does NOT
    # repoint data loading: src.data.get_data_list always walks exactly
    # root_dir/0/ and root_dir/1/, independent of this list's contents or
    # length. dlorchestra's training/eval metrics (F2/AUC/checkpointing)
    # also assume binary classification regardless of num_classes. In
    # short: this isn't a supported way to go multi-class, just the
    # class-label strings for the fixed binary 0/1 folder layout.
    "CLASSES": ["0", "1"],
}

# Allowed values for keys that only accept one of a fixed set of strings.
_ALLOWED_VALUES: dict[str, set] = {
    "normalization": {"percentile", "zscore", "histogram", "none"},
    "balancing": {"weighted_cost_function", "oversampling", "none"},
    "LOSS_FN": {"cross_entropy", "focal"},
    "CHECKPOINT_METRIC": {"val_f2", "val_auc"},
    "LR_SCHEDULER": {"cosine", "none"},
    "AMP_DTYPE": {"bfloat16", "float16"},
}

_REQUIRED_KEYS = ("name", "model", "data_path_key")

# "_base_name" is injected at runtime under SLURM; "_comment" is a JSON
# doesn't-support-comments convention used in some experiment files.
_EXEMPT_UNKNOWN_KEYS = ("_base_name", "_comment")


def _check_histogram_template(config: dict) -> str | None:
    if config.get("normalization") != "histogram":
        return None
    template = config.get("histogram_template")
    if not template or not Path(template).exists():
        return f"normalization='histogram' requires an existing histogram_template (got {template!r})"
    return None


def _check_data_path_key(config: dict) -> str | None:
    data_path_key = config.get("data_path_key")
    if data_path_key is not None and data_path_key not in DATASET_PATHS:
        return f"data_path_key={data_path_key!r} not in DATASET_PATHS: {sorted(DATASET_PATHS)}"
    return None


def _check_model(config: dict) -> str | None:
    model = config.get("model")
    if model is None:
        return None
    from src.models import SUPPORTED_MODELS
    if model not in SUPPORTED_MODELS:
        return f"model={model!r} not in SUPPORTED_MODELS: {sorted(SUPPORTED_MODELS)}"
    return None


def validate_config(config: dict) -> None:
    """Sanity-check a merged experiment config. Raises ValueError listing all problems found.

    Catches the mistakes that are cheap to catch before burning GPU time:
    missing required keys, unknown keys (typos), bad enum values, and an
    unregistered model or dataset key.

    Generic mechanics (required keys, unknown-key/typo detection, enum
    checks) live in dlorchestra.config; the checks below are project-specific
    (they touch this project's dataset paths and model registry).
    """
    from dlorchestra.config import validate_config as _dl_validate_config

    _dl_validate_config(
        config,
        default_config=DEFAULT_CONFIG,
        required_keys=_REQUIRED_KEYS,
        allowed_values=_ALLOWED_VALUES,
        extra_checks=(_check_histogram_template, _check_data_path_key, _check_model),
        exempt_unknown_keys=_EXEMPT_UNKNOWN_KEYS,
    )
