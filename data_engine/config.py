import os
from pathlib import Path

# Adjust DATA_ROOT to wherever your data lives.
# Override with DATA_ROOT env var for cluster use.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/data"))

# Exams with these class values are excluded from QC and conversion.
# Class 2 is excluded by default (not part of the training cohort).
EXCLUDE_CLASSES: set[int] = {2}

SOURCE_DIR        = Path(os.environ.get("SOURCE_DIR", DATA_ROOT / "zips"))         # input .zip archives
STAGING_ROOT      = Path(os.environ.get("STAGING_ROOT", DATA_ROOT / "raw"))       # extracted raw data
NIFTI_ROOT        = Path(os.environ.get("NIFTI_ROOT", DATA_ROOT / "nifti"))       # converted NIfTI files (full dataset)
DATASETS_ROOT     = Path(os.environ.get("DATASETS_ROOT", DATA_ROOT / "datasets")) # resampled datasets (one dir per variant)
REPORTS_DIR  = Path(__file__).parent / "reports"   # CSV outputs

DATASET_CSV = Path(__file__).parent.parent / "dataset" / "dataset.csv"

STEP0_REPORT         = REPORTS_DIR / "step0_discover.csv"
STEP1_REPORT         = REPORTS_DIR / "step1_convert.csv"
MIXED_SERIES_REPORT  = REPORTS_DIR / "mixed_series_report.csv"
STEP2_REPORT         = REPORTS_DIR / "step2_resample.csv"

# Aliases — downstream scripts that read the pipeline/datasets report by old name.
PIPELINE_REPORT = STEP1_REPORT
DATASETS_REPORT = STEP2_REPORT


# ---------------------------------------------------------------------------
# Test mode — pipeline run on sample data, outputs under DATA_ROOT
# ---------------------------------------------------------------------------

SAMPLE_SOURCE_DIR = DATA_ROOT / "sample_zips"   # sample zip archives


def enable_test_mode() -> None:
    """Redirect all data paths to sample_* dirs under DATA_ROOT.

    Mutates attributes on this module's own entry in sys.modules (the
    `import config` below re-fetches that same cached module object, not a
    fresh copy) so every other module that already did `import config`
    sees the change too. Must be called once, early — before any other
    module has read the attributes being overwritten here into a local
    variable (those would keep the pre-test-mode value).
    """
    import config

    config.SOURCE_DIR    = config.SAMPLE_SOURCE_DIR
    config.STAGING_ROOT  = DATA_ROOT / "sample_raw"
    config.NIFTI_ROOT    = DATA_ROOT / "sample_nifti"
    config.DATASETS_ROOT = DATA_ROOT / "datasets"        # shared root; variants get sample_ prefix
    config.REPORTS_DIR   = Path(__file__).parent / "reports" / "sample"

    config.STEP0_REPORT        = config.REPORTS_DIR / "step0_discover.csv"
    config.STEP1_REPORT        = config.REPORTS_DIR / "step1_convert.csv"
    config.MIXED_SERIES_REPORT = config.REPORTS_DIR / "mixed_series_report.csv"
    config.STEP2_REPORT        = config.REPORTS_DIR / "step2_resample.csv"
    config.PIPELINE_REPORT     = config.STEP1_REPORT
    config.DATASETS_REPORT     = config.STEP2_REPORT

    # Prefix variant names so output lands at datasets/sample_iso_128, etc.
    config.VARIANTS = [
        {**v, "name": f"sample_{v['name']}"} for v in config.VARIANTS
    ]

# Stage 6 — centroid crop target (applied at conversion time, before saving NIfTI)
TARGET_FOV_MM = 180.0   # XY FOV to crop to; exams already ≤ this are untouched

# Stage 5 — QC thresholds
EXTENT_MIN_MM            = 90.0   # total z-extent below this → extent_issue = "too_short"
EXTENT_MAX_MM            = 300.0  # total z-extent above this → extent_issue = "too_long"
SPACING_STD_TOLERANCE_MM = 0.1    # std of inter-slice distances above this → inconsistent_spacing

# ---------------------------------------------------------------------------
# Resampler — dataset variants
# ---------------------------------------------------------------------------

# Two datasets: isotropic cubic 128³ and 256³.
# Spacing = TARGET_FOV_MM / dim (all NIfTIs are at 180mm XY after centroid crop).
# Cropping is handled by data augmentation (random zoom) at training time, not here.

VARIANTS: list[dict] = [
    # iso_128: 128³ cubic, 1.40625 mm/voxel (180mm / 128)
    {"name": "iso_128", "strategy": "iso_fixed", "output_shape": (128, 128, 128), "target_spacing_mm": (1.40625, 1.40625, 1.40625)},
    # iso_256: 256³ cubic, 0.703125 mm/voxel (180mm / 256)
    {"name": "iso_256", "strategy": "iso_fixed", "output_shape": (256, 256, 256), "target_spacing_mm": (0.703125, 0.703125, 0.703125)},
    # aniso_384: 384×384×128, XY=0.46875 mm (180/384), Z=1.40625 mm (180/128)
    # 3:1 anisotropy ratio, no upsampling needed (384 = smallest post-crop XY matrix)
    {"name": "aniso_384", "strategy": "iso_fixed", "output_shape": (384, 384, 128), "target_spacing_mm": (0.46875, 0.46875, 1.40625)},
    # aniso_512: 512×512×128, XY=0.351563 mm (180/512), Z=1.40625 mm (180/128)
    # 4:1 anisotropy ratio, mild 1.33x upsample for 22% of exams (groups 3,5)
    {"name": "aniso_512", "strategy": "iso_fixed", "output_shape": (512, 512, 128), "target_spacing_mm": (0.351563, 0.351563, 1.40625)},
    # --- Z-cropped variants (center-crop Z to ~90mm, keep 180mm XY FOV) ---
    # c90z_256: 256×256×128, 0.703125 mm isotropic (180/256 XY, 90/128 Z)
    # Same XY resolution as iso_256 but Z is center-cropped to 90mm → no Z padding waste
    {"name": "c90z_256", "strategy": "iso_fixed", "output_shape": (256, 256, 128), "target_spacing_mm": (0.703125, 0.703125, 0.703125)},
    # c90z_384: 384×384×192, 0.46875 mm isotropic (180/384 XY, 90/192 Z)
    # Same XY resolution as aniso_384 but isotropic; Z upsamples most exams (max 1.6x for 0.75mm group)
    {"name": "c90z_384", "strategy": "iso_fixed", "output_shape": (384, 384, 192), "target_spacing_mm": (0.46875, 0.46875, 0.46875)},
]
