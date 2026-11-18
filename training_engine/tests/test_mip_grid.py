import numpy as np
import pytest

import config
from src.data import LoadNpyd


def test_grid_windows_match_generator():
    # cross-engine check: needs data_engine importable (installed in this venv or on PYTHONPATH)
    grid = pytest.importorskip("steps.step_mip.generate_mip_grid")
    assert config.MIP_GRID_WINDOWS == grid.WINDOWS


def test_input_channels_selects_axial(tmp_path):
    arr = np.stack([np.full((4, 4), c, np.float16) for c in range(3)])
    np.save(tmp_path / "x.npy", arr)
    out = LoadNpyd(keys=["image"], channels=[0])({"image": tmp_path / "x.npy"})["image"]
    assert out.shape == (1, 4, 4) and out.dtype == np.float32 and out.max() == 0
