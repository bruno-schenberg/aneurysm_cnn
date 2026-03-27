import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from src.nifti_resize import (
    TARGET_SHAPE,
    generate_variant_a,
    generate_variant_b,
    generate_variant_c,
    generate_variant_d,
    generate_variant_e,
)

ALL_VARIANTS = [
    generate_variant_a,
    generate_variant_b,
    generate_variant_c,
    generate_variant_d,
    generate_variant_e,
]


@pytest.fixture
def make_nifti(tmp_path):
    """Return a factory that writes a synthetic NIfTI filled with ones.

    The affine is a diagonal matrix built from `zooms` so that
    ``header.get_zooms()[:3]`` equals the supplied voxel sizes.
    """
    counter = [0]

    def _make(shape, zooms):
        data = np.ones(shape, dtype=np.float32)
        affine = np.diag([float(zooms[0]), float(zooms[1]), float(zooms[2]), 1.0])
        img = nib.Nifti1Image(data, affine)
        counter[0] += 1
        path = tmp_path / f"input_{counter[0]}.nii.gz"
        nib.save(img, path)
        return path

    return _make


# ---------------------------------------------------------------------------
# US1 — shape and spacing (T012)
# ---------------------------------------------------------------------------


def test_variant_a_shape_and_spacing(make_nifti, tmp_path):
    """Variant A output must be 128³ at 1 mm isotropic spacing."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_a.nii.gz"
    generate_variant_a(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    assert np.allclose(zooms, (1.0, 1.0, 1.0), atol=1e-3)


def test_variant_b_shape_and_spacing(make_nifti, tmp_path):
    """Variant B output must be 128³ with nominal 1 mm isotropic spacing."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_b.nii.gz"
    generate_variant_b(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    assert np.allclose(zooms, (1.0, 1.0, 1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# US1 — shape only (T013)
# ---------------------------------------------------------------------------


def test_variant_c_shape(make_nifti, tmp_path):
    path = make_nifti((200, 200, 150), (1.0, 1.0, 1.0))
    out = tmp_path / "out_c.nii.gz"
    generate_variant_c(path, out)
    assert nib.load(out).shape == TARGET_SHAPE


def test_variant_d_shape(make_nifti, tmp_path):
    path = make_nifti((200, 200, 150), (1.0, 1.0, 1.0))
    out = tmp_path / "out_d.nii.gz"
    generate_variant_d(path, out)
    assert nib.load(out).shape == TARGET_SHAPE


def test_variant_e_shape(make_nifti, tmp_path):
    path = make_nifti((200, 200, 150), (1.0, 1.0, 1.0))
    out = tmp_path / "out_e.nii.gz"
    generate_variant_e(path, out)
    assert nib.load(out).shape == TARGET_SHAPE


def test_all_variants_no_exception_on_valid_input(make_nifti, tmp_path):
    """All five variants complete without raising on a standard input."""
    path = make_nifti((200, 200, 150), (1.0, 1.0, 1.0))
    for i, fn in enumerate(ALL_VARIANTS):
        out = tmp_path / f"out_{i}.nii.gz"
        fn(path, out)
        assert out.exists(), f"{fn.__name__} did not produce an output file"


# ---------------------------------------------------------------------------
# US1 — small input padding (T014)
# ---------------------------------------------------------------------------


def test_variant_c_small_input_is_padded(make_nifti, tmp_path):
    """Variant C must pad a (64³) input; padded region must contain zeros."""
    path = make_nifti((64, 64, 64), (1.0, 1.0, 1.0))
    out = tmp_path / "out_c_small.nii.gz"
    generate_variant_c(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # The first voxel slice must be padded (zero), not original data (one)
    assert data[0, :, :].max() == 0.0


def test_variant_a_small_input_is_padded(make_nifti, tmp_path):
    """Variant A must pad after resampling when input is smaller than 128³."""
    path = make_nifti((64, 64, 64), (1.0, 1.0, 1.0))
    out = tmp_path / "out_a_small.nii.gz"
    generate_variant_a(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    assert data[0, :, :].max() == 0.0


def test_variant_e_small_input_is_padded(make_nifti, tmp_path):
    """Variant E must pad when the scaled volume is smaller than 128³ on some axis.

    A cubic (64,64,64) input at 1mm is uniformly scaled to (128,128,128), leaving
    nothing to pad. Use a non-cubic (64,32,64) input instead: after uniform scaling
    (scale=2.0) it becomes (128,64,128), so the y-axis requires zero-padding to 128.
    """
    path = make_nifti((64, 32, 64), (1.0, 1.0, 1.0))
    out = tmp_path / "out_e_small.nii.gz"
    generate_variant_e(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # y-axis was padded from 64 to 128; edge slices must be zero
    assert data[:, 0, :].max() == 0.0
    assert data[:, -1, :].max() == 0.0


# ---------------------------------------------------------------------------
# US1 — identity and large volume (T015)
# ---------------------------------------------------------------------------


def test_variant_c_identity_if_already_128(make_nifti, tmp_path):
    """Variant C must leave a 128³ volume unchanged."""
    path = make_nifti((128, 128, 128), (1.0, 1.0, 1.0))
    out = tmp_path / "out_c_id.nii.gz"
    generate_variant_c(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    assert np.allclose(data, 1.0)


def test_variant_d_identity_if_already_128(make_nifti, tmp_path):
    """Variant D must leave a 128³ volume unchanged (up to interpolation noise)."""
    path = make_nifti((128, 128, 128), (1.0, 1.0, 1.0))
    out = tmp_path / "out_d_id.nii.gz"
    generate_variant_d(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    assert np.allclose(data, 1.0, atol=1e-5)


def test_variant_d_large_volume(make_nifti, tmp_path):
    """Variant D must handle a large anisotropic volume without crashing."""
    path = make_nifti((400, 400, 180), (0.5, 0.5, 1.0))
    out = tmp_path / "out_d_large.nii.gz"
    generate_variant_d(path, out)
    assert nib.load(out).shape == TARGET_SHAPE


# ---------------------------------------------------------------------------
# US1 — variant E largest dim (T016)
# ---------------------------------------------------------------------------


def test_variant_e_largest_dim_not_exceeded(make_nifti, tmp_path):
    """Variant E must scale so max(scaled_shape) == 128 before the pad step.

    For input (300, 200, 150) at (0.5, 0.5, 1.0) mm the isotropic volume is
    ~(300, 200, 300) at 0.5 mm. After scaling, x and z map to 128 and y maps
    to ~85. The pad step then fills y from 85 to 128. Edge slices in y must
    therefore be zero (padded), confirming no axis exceeded 128 before padding.
    """
    path = make_nifti((300, 200, 150), (0.5, 0.5, 1.0))
    out = tmp_path / "out_e_large.nii.gz"
    generate_variant_e(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # y-axis edge slices must be padded zeros because y was < 128 before padding
    assert data[:, 0, :].max() == 0.0
    assert data[:, -1, :].max() == 0.0


# ---------------------------------------------------------------------------
# US2 — orientation preserved (T017)
# ---------------------------------------------------------------------------


def test_all_variants_orientation_preserved(make_nifti, tmp_path):
    """No variant may flip or mirror any axis (FR-011)."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    input_affine = np.diag([0.8, 0.8, 1.2, 1.0])
    input_axcodes = nib.orientations.aff2axcodes(input_affine)

    for i, fn in enumerate(ALL_VARIANTS):
        out = tmp_path / f"out_orient_{i}.nii.gz"
        fn(path, out)
        output_affine = nib.load(out).affine
        output_axcodes = nib.orientations.aff2axcodes(output_affine)
        assert output_axcodes == input_axcodes, (
            f"{fn.__name__}: expected axcodes {input_axcodes}, got {output_axcodes}"
        )


# ---------------------------------------------------------------------------
# US2 — no NaN or Inf, nonzero output (T018)
# ---------------------------------------------------------------------------


def test_all_variants_no_nan_or_inf(make_nifti, tmp_path):
    """No variant may produce NaN or Inf voxels (FR-012)."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    for i, fn in enumerate(ALL_VARIANTS):
        out = tmp_path / f"out_nan_{i}.nii.gz"
        fn(path, out)
        data = nib.load(out).get_fdata(dtype=np.float32)
        assert np.isnan(data).sum() == 0, f"{fn.__name__} produced NaN voxels"
        assert np.isinf(data).sum() == 0, f"{fn.__name__} produced Inf voxels"


def test_all_variants_nonzero_output(make_nifti, tmp_path):
    """Brain signal must be present in the output window of every variant (FR-012)."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    for i, fn in enumerate(ALL_VARIANTS):
        out = tmp_path / f"out_nonzero_{i}.nii.gz"
        fn(path, out)
        data = nib.load(out).get_fdata(dtype=np.float32)
        assert data.max() > 0, f"{fn.__name__} produced an all-zero output"


# ---------------------------------------------------------------------------
# US2 — header–affine consistency (T019)
# ---------------------------------------------------------------------------


def test_ab_variants_header_affine_consistency(make_nifti, tmp_path):
    """Variants A and B must report 1 mm spacing in both header and affine (FR-013)."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    for i, fn in enumerate([generate_variant_a, generate_variant_b]):
        out = tmp_path / f"out_ab_hdr_{i}.nii.gz"
        fn(path, out)
        img = nib.load(out)
        header_zooms = np.array(img.header.get_zooms()[:3])
        affine_zooms = np.abs(np.diagonal(img.affine)[:3])
        assert np.allclose(header_zooms, affine_zooms, atol=1e-4), (
            f"{fn.__name__}: header zooms {header_zooms} != affine zooms {affine_zooms}"
        )
        assert np.allclose(header_zooms, 1.0, atol=1e-3), (
            f"{fn.__name__}: expected 1 mm zooms, got {header_zooms}"
        )


def test_all_variants_header_affine_consistency(make_nifti, tmp_path):
    """Header zooms and affine diagonal must agree for all variants (FR-013)."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    for i, fn in enumerate(ALL_VARIANTS):
        out = tmp_path / f"out_hdr_{i}.nii.gz"
        fn(path, out)
        img = nib.load(out)
        header_zooms = np.array(img.header.get_zooms()[:3])
        affine_zooms = np.abs(np.diagonal(img.affine)[:3])
        assert np.allclose(header_zooms, affine_zooms, atol=1e-4), (
            f"{fn.__name__}: header zooms {header_zooms} != affine zooms {affine_zooms}"
        )
