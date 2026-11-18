from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from steps.step_mip.generate_mip_grid import WINDOWS, fit_view, process_volume, window_bounds


@pytest.mark.parametrize("window,expected", [
    ("top60", (90.0, 150.0, False)),
    ("center90", (30.0, 120.0, False)),
    ("vessel60", (90.0, 150.0, True)),   # anchor 140 runs off the top -> shifted down
    ("top120", (30.0, 150.0, False)),
    ("pct90", (15.0, 150.0, False)),
    ("full", (0.0, 150.0, False)),
])
def test_window_bounds(window, expected):
    assert window_bounds(window, extent=150.0, anchor=140.0) == pytest.approx(expected)


def test_window_longer_than_volume_is_truncated():
    assert window_bounds("top120", extent=100.0, anchor=50.0) == (0.0, 100.0, True)


def test_pad_keeps_mm_aspect():
    out = fit_view(np.ones((10, 5), np.float32), (180.0, 90.0), 64, "pad")
    assert out.shape == (64, 64)
    assert out[:, 32].sum() == 64 and out[32, :].sum() == 32  # 180x90 mm -> 64x32 px


def _save(tmp_path, name, data, affine):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_orientation_independent(tmp_path):
    vol = np.zeros((40, 40, 60), np.float32)
    vol[10, 20, 55] = 100.0  # bright voxel near the vertex
    lps = np.diag([-1.0, -1.0, 2.0, 1.0])
    ras = np.diag([1.0, 1.0, 2.0, 1.0])
    args = SimpleNamespace(windows=list(WINDOWS), aspects=["stretch", "pad"], sizes=[32],
                           crop_fov_mm=None, n4=False, mask_dilate_mm=10.0)
    a, rows = process_volume(_save(tmp_path, "lps.nii.gz", vol, lps), None, args)
    b, _ = process_volume(_save(tmp_path, "ras.nii.gz", vol[::-1, ::-1], ras), None, args)
    assert set(a) == {f"{w}_{asp}_32" for w in WINDOWS for asp in ("stretch", "pad")}
    for k in a:
        assert a[k].shape == (3, 32, 32) and a[k].dtype == np.float16
        np.testing.assert_array_equal(a[k], b[k])
    assert a["top60_stretch_32"].max() > 0          # top window sees the voxel
    assert a["center90_stretch_32"][1].max() == 0   # 120 mm slab, centre 90 mm misses z=110 mm
    assert {r["window"] for r in rows} == set(WINDOWS)


def test_mask_dilation(tmp_path):
    vol = np.zeros((40, 40, 60), np.float32)
    vol[30, 20, 30] = 100.0  # 10 mm outside the mask below (1 mm voxels in X)
    mask = np.zeros_like(vol)
    mask[5:20, 5:35, 5:55] = 1
    aff = np.diag([-1.0, -1.0, 1.0, 1.0])
    vol_p, mask_p = _save(tmp_path, "v.nii.gz", vol, aff), _save(tmp_path, "m.nii.gz", mask, aff)
    base = dict(windows=["full"], aspects=["stretch"], sizes=[32], crop_fov_mm=None, n4=False)
    cut, _ = process_volume(vol_p, mask_p, SimpleNamespace(**base, mask_dilate_mm=0.0))
    kept, _ = process_volume(vol_p, mask_p, SimpleNamespace(**base, mask_dilate_mm=15.0))
    assert cut["full_stretch_32"].max() == 0 and kept["full_stretch_32"].max() > 0
