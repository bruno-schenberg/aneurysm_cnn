"""
conftest.py

Shared pytest fixtures for the training_engine test suite.

The SAMPLE_DATASET_D constant points to the pre-built 12-case variant-D
dataset (native resolution → 128×128×128 trilinear shrink, 6 negative /
6 positive) stored at a well-known path on the development machine and HPC.

Tests that require real NIfTI files depend on fixtures defined here.  Each
fixture calls ``pytest.skip`` automatically when the dataset directory is
not mounted, so the same test suite runs cleanly on machines that do not
have the external dataset (e.g. a fresh CI runner).  Tests that use only
synthetic in-memory data are unaffected.
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

SAMPLE_DATASET_D = Path("/mnt/data/nifti-sample-dataset-D")
"""
Root directory of the 12-case variant-D sample dataset used by integration
tests.  Layout mirrors the production dataset format expected by
``get_data_list``:

    SAMPLE_DATASET_D/
        0/   ← 6 negative cases (no aneurysm), 128×128×128 float32 NIfTI
        1/   ← 6 positive cases (aneurysm present), 128×128×128 float32 NIfTI
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_dataset_path() -> Path:
    """
    Root directory of the 12-case sample dataset (variant D).

    Session-scoped so the path check runs only once per test session.
    Skips automatically when the dataset is not mounted.
    """
    if not SAMPLE_DATASET_D.exists():
        pytest.skip(
            f"Sample dataset not found at {SAMPLE_DATASET_D}. "
            "Mount the dataset or run only unit tests with: pytest -m 'not integration'"
        )
    return SAMPLE_DATASET_D


@pytest.fixture(scope="session")
def sample_nifti_neg(sample_dataset_path: Path) -> Path:
    """
    Path to the first alphabetically sorted negative-class NIfTI in the
    sample dataset.  Used by transform tests that need a real file on disk
    without caring about the specific case.
    """
    files = sorted((sample_dataset_path / "0").glob("*.nii.gz"))
    assert files, f"No .nii.gz files found in {sample_dataset_path / '0'}"
    return files[0]


@pytest.fixture(scope="session")
def sample_nifti_pos(sample_dataset_path: Path) -> Path:
    """
    Path to the first alphabetically sorted positive-class NIfTI in the
    sample dataset.  Paired with ``sample_nifti_neg`` when both classes
    are needed.
    """
    files = sorted((sample_dataset_path / "1").glob("*.nii.gz"))
    assert files, f"No .nii.gz files found in {sample_dataset_path / '1'}"
    return files[0]
