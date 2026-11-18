"""scripts/check_datasets.py's per-file checks catch broken arrays."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_datasets import check_file  # noqa: E402

GOOD = np.random.default_rng(0).random((3, 8, 8)).astype(np.float16)


@pytest.mark.parametrize("name,arr,expect", [
    ("good", GOOD, []),
    ("empty", np.zeros((3, 8, 8), np.float16), ["constant"]),
    ("blank_view", np.concatenate([GOOD[:2], np.zeros((1, 8, 8), np.float16)]), ["view 2 constant"]),
    ("nan", np.where(GOOD > 0.5, np.nan, GOOD).astype(np.float16), ["non-finite values"]),
])
def test_check_file(tmp_path, name, arr, expect):
    np.save(tmp_path / f"{name}.npy", arr)
    _, shape, bad, _ = check_file(str(tmp_path / f"{name}.npy"))
    assert shape == ((3, 8, 8), "float16") and bad == expect


def test_unreadable(tmp_path):
    (tmp_path / "x.npy").write_bytes(b"")
    assert check_file(str(tmp_path / "x.npy"))[2][0].startswith("load failed")
