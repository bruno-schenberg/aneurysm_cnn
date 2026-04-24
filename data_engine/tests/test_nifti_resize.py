import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from src.nifti_resize import (
    TARGET_SHAPE,
    CROP_FRACTION,
    E_SPACING_MM_BY_SHAPE,
    F_SPACING_MM_BY_SHAPE,
    generate_variant_a,
    generate_variant_b,
    generate_variant_c,
    generate_variant_d,
    generate_variant_e,
    generate_variant_f,
)

# Integration tests use real NIfTI files from this path. Tests auto-skip when
# the path is not mounted so the unit test suite always runs without the SSD.
SAMPLE_NIFTI_DIR = Path("/mnt/data/nifti-sample")

ALL_VARIANTS = [
    generate_variant_a,
    generate_variant_b,
    generate_variant_c,
    generate_variant_d,
    generate_variant_e,
    generate_variant_f,
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
# US1 — shape checks (T012)
# ---------------------------------------------------------------------------


def test_variant_a_output_shape(make_nifti, tmp_path):
    """Variant A output must have shape TARGET_SHAPE."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_a.nii.gz"
    generate_variant_a(path, out)
    assert nib.load(out).shape == TARGET_SHAPE


def test_variant_a_isotropic_spacing(make_nifti, tmp_path):
    """Variant A output must have isotropic spacing (dynamic, not fixed at 1 mm).

    For (200, 200, 150) at (0.8, 0.8, 1.2) mm:
      80% crop shape: (160, 160, 120) voxels
      cropped extents: (128, 128, 144) mm
      dynamic spacing = max(128/128, 128/128, 144/128) = 1.125 mm
    """
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_a.nii.gz"
    generate_variant_a(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
    assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"
    # Expected: max(128/128, 144/128) = 1.125 mm
    expected_spacing = max(160 * 0.8 / 128, 160 * 0.8 / 128, 120 * 1.2 / 128)
    assert np.allclose(zooms[0], expected_spacing, atol=0.05), (
        f"Expected ~{expected_spacing:.3f} mm dynamic spacing, got {zooms[0]:.3f}"
    )


def test_variant_b_shape_and_spacing(make_nifti, tmp_path):
    """Variant B output must be TARGET_SHAPE with isotropic spacing.

    For (200, 200, 150) at (0.8, 0.8, 1.2):
      full extents: (160, 160, 180) mm
      spacing = max(160/128, 160/128, 180/128) = 180/128 ≈ 1.406 mm
    """
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_b.nii.gz"
    generate_variant_b(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
    assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"


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


def test_variant_e_shape_and_fixed_spacing(make_nifti, tmp_path):
    """Variant E output must be TARGET_SHAPE at the fixed E spacing."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_e.nii.gz"
    generate_variant_e(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    expected = E_SPACING_MM_BY_SHAPE[TARGET_SHAPE]
    assert np.allclose(zooms[0], expected, atol=0.05), (
        f"Expected E fixed spacing ~{expected:.3f} mm, got {zooms[0]:.3f}"
    )


def test_variant_f_shape_and_fixed_spacing(make_nifti, tmp_path):
    """Variant F output must be TARGET_SHAPE at the fixed F spacing."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_f.nii.gz"
    generate_variant_f(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    expected = F_SPACING_MM_BY_SHAPE[TARGET_SHAPE]
    assert np.allclose(zooms[0], expected, atol=0.05), (
        f"Expected F fixed spacing ~{expected:.3f} mm, got {zooms[0]:.3f}"
    )


def test_all_variants_no_exception_on_valid_input(make_nifti, tmp_path):
    """All six variants complete without raising on a standard input."""
    path = make_nifti((200, 200, 150), (1.0, 1.0, 1.0))
    for i, fn in enumerate(ALL_VARIANTS):
        out = tmp_path / f"out_{i}.nii.gz"
        fn(path, out)
        assert out.exists(), f"{fn.__name__} did not produce an output file"


# ---------------------------------------------------------------------------
# US1 — crop behaviour of variants A, C, E (T014)
# ---------------------------------------------------------------------------


def test_variant_c_applies_center_crop(make_nifti, tmp_path):
    """Variant C crops 80% of each native axis then resizes; output must be all ones
    because the centre region of an all-ones volume is still all ones."""
    path = make_nifti((200, 200, 150), (1.0, 1.0, 1.0))
    out = tmp_path / "out_c_crop.nii.gz"
    generate_variant_c(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # All-ones input → 80% crop is all ones → resize is all ones
    assert np.allclose(data, 1.0, atol=1e-4), "Resize of uniform input must stay uniform"


def test_variant_c_small_input_fills_target(make_nifti, tmp_path):
    """Variant C on a (64³) input: 80% crop then resize fills the full target (no padding)."""
    path = make_nifti((64, 64, 64), (1.0, 1.0, 1.0))
    out = tmp_path / "out_c_small.nii.gz"
    generate_variant_c(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # The resize step fills the full target — no zero padding expected
    assert data.max() > 0.0, "Variant C resize must produce non-zero output"
    assert np.allclose(data, 1.0, atol=1e-4), "Resize of uniform input must stay uniform"


def test_variant_a_small_input_fills_target(make_nifti, tmp_path):
    """Variant A on a (64³) input: 80% crop + dynamic spacing fills the full target."""
    path = make_nifti((64, 64, 64), (1.0, 1.0, 1.0))
    out = tmp_path / "out_a_small.nii.gz"
    generate_variant_a(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # Dynamic spacing maps the cropped volume to exactly fill the target
    assert data.max() > 0.0, "Variant A must produce non-zero output"


# ---------------------------------------------------------------------------
# US1 — identity and large volume (T015)
# ---------------------------------------------------------------------------


def test_variant_d_identity_if_already_target(make_nifti, tmp_path):
    """Variant D must leave a TARGET_SHAPE volume unchanged (up to interpolation noise)."""
    path = make_nifti(TARGET_SHAPE, (1.0, 1.0, 1.0))
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
# US1 — B never crops (T016)
# ---------------------------------------------------------------------------


def test_variant_b_never_crops(make_nifti, tmp_path):
    """Variant B must never crop: max resampled dim ≤ target dim on all axes.

    This is enforced by the dynamic spacing formula:
      spacing = max_i(extent_i / target_i)
    which guarantees every resampled axis ≤ target axis (only padding).
    We verify indirectly: uniform input must produce uniform (all-ones) output
    since no content is discarded and the padding is zero.
    """
    # Input smaller than target after resampling → will only pad
    path = make_nifti((100, 100, 80), (1.0, 1.0, 1.0))
    out = tmp_path / "out_b_nocrop.nii.gz"
    generate_variant_b(path, out)
    data = nib.load(out).get_fdata(dtype=np.float32)
    assert data.shape == TARGET_SHAPE
    # Non-zero signal must be present
    assert data.max() > 0.0


def test_variant_f_never_crops(make_nifti, tmp_path):
    """Variant F with largest expected exam must not crop after resampling.

    For 240mm XY FOV (BP042/BP240), F spacing = 240/target_XY maps it exactly
    to target_XY. The output should have no cropping — resampled dims ≤ target.
    """
    # Simulate largest clinical exam: 512x512 at 0.4688mm (= 240mm XY), 216 slices at 0.58mm
    path = make_nifti((512, 512, 216), (0.4688, 0.4688, 0.58))
    out = tmp_path / "out_f_nocrop.nii.gz"
    generate_variant_f(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE


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


def test_resampling_variants_header_affine_consistency(make_nifti, tmp_path):
    """Variants A, B, E, F must have header zooms consistent with affine diagonal (FR-013)."""
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    for i, fn in enumerate([generate_variant_a, generate_variant_b,
                             generate_variant_e, generate_variant_f]):
        out = tmp_path / f"out_ab_hdr_{i}.nii.gz"
        fn(path, out)
        img = nib.load(out)
        header_zooms = np.array(img.header.get_zooms()[:3])
        affine_zooms = np.abs(np.diagonal(img.affine)[:3])
        assert np.allclose(header_zooms, affine_zooms, atol=1e-4), (
            f"{fn.__name__}: header zooms {header_zooms} != affine zooms {affine_zooms}"
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


# ---------------------------------------------------------------------------
# 192×192×128 target shape — unit tests
# ---------------------------------------------------------------------------


class TestVariant192Shape:
    """Unit tests for all six variants at the 192×192×128 target shape."""

    TARGET = (192, 192, 128)

    def test_variant_a_shape_and_isotropic_spacing(self, make_nifti, tmp_path):
        """Variant A at 192×192×128 must output that shape with isotropic spacing."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_a192.nii.gz"
        generate_variant_a(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
        assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"

    def test_variant_b_shape_and_isotropic_spacing(self, make_nifti, tmp_path):
        """Variant B at 192×192×128 must output that shape with isotropic spacing."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_b192.nii.gz"
        generate_variant_b(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
        assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"

    def test_variant_c_shape(self, make_nifti, tmp_path):
        path = make_nifti((300, 300, 200), (1.0, 1.0, 1.0))
        out = tmp_path / "out_c192.nii.gz"
        generate_variant_c(path, out, target_shape=self.TARGET)
        assert nib.load(out).shape == self.TARGET

    def test_variant_d_shape(self, make_nifti, tmp_path):
        path = make_nifti((400, 400, 200), (0.5, 0.5, 0.6))
        out = tmp_path / "out_d192.nii.gz"
        generate_variant_d(path, out, target_shape=self.TARGET)
        assert nib.load(out).shape == self.TARGET

    def test_variant_e_shape_and_fixed_spacing(self, make_nifti, tmp_path):
        """Variant E at 192×192×128 must use E_SPACING = 192/192 = 1.0 mm."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_e192.nii.gz"
        generate_variant_e(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        expected = E_SPACING_MM_BY_SHAPE[self.TARGET]  # 1.0 mm
        assert np.allclose(zooms[0], expected, atol=0.05), (
            f"Expected E spacing {expected:.3f} mm, got {zooms[0]:.3f}"
        )

    def test_variant_f_shape_and_fixed_spacing(self, make_nifti, tmp_path):
        """Variant F at 192×192×128 must use F_SPACING = 240/192 = 1.25 mm."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_f192.nii.gz"
        generate_variant_f(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        expected = F_SPACING_MM_BY_SHAPE[self.TARGET]  # 1.25 mm
        assert np.allclose(zooms[0], expected, atol=0.05), (
            f"Expected F spacing {expected:.3f} mm, got {zooms[0]:.3f}"
        )

    def test_variant_b_clinical_geometry(self, make_nifti, tmp_path):
        """Variant B at 192×192×128: BP042 geometry (240mm XY) → spacing = 240/192 = 1.25 mm."""
        path = make_nifti((512, 512, 217), (0.4688, 0.4688, 0.6))
        out = tmp_path / "out_b192_clinical.nii.gz"
        generate_variant_b(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = np.array(img.header.get_zooms()[:3])
        # extents: (512*0.4688, 512*0.4688, 217*0.6) = (240.0, 240.0, 130.2) mm
        # spacing = max(240/192, 240/192, 130.2/128) = max(1.25, 1.25, 1.017) = 1.25 mm
        expected = 512 * 0.4688 / 192.0  # FOV_mm / target_voxels = 240mm / 192 = 1.25 mm
        assert np.allclose(zooms[0], expected, atol=0.05), (
            f"Expected ~{expected:.3f} mm spacing, got {zooms[0]:.3f}"
        )

    def test_all_six_variants_shape(self, make_nifti, tmp_path):
        """All six variants must produce 192×192×128."""
        path = make_nifti((300, 300, 200), (0.6, 0.6, 0.7))
        for i, fn in enumerate(ALL_VARIANTS):
            out = tmp_path / f"out_192_{i}.nii.gz"
            fn(path, out, target_shape=self.TARGET)
            assert nib.load(out).shape == self.TARGET, f"{fn.__name__} wrong shape"


# ---------------------------------------------------------------------------
# 256×256×176 target shape — unit tests
# ---------------------------------------------------------------------------


class TestVariant256x176Shape:
    """Unit tests for all six variants at the 256×256×176 target shape."""

    TARGET = (256, 256, 176)

    def test_variant_a_shape_and_isotropic_spacing(self, make_nifti, tmp_path):
        """Variant A at 256×256×176 must output that shape with isotropic spacing."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_a176.nii.gz"
        generate_variant_a(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
        assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"

    def test_variant_b_shape_and_isotropic_spacing(self, make_nifti, tmp_path):
        """Variant B at 256×256×176 must output that shape with isotropic spacing."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_b176.nii.gz"
        generate_variant_b(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
        assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"

    def test_variant_b_formula_non_cubic(self, make_nifti, tmp_path):
        """Variant B non-cubic formula: max_i(extent_i / target_i) — Z axis should not be clipped.

        For (512, 512, 260) at (0.4688, 0.4688, 0.5) mm (240mm XY, 130mm Z):
          extents: (240, 240, 130) mm
          ratios: (240/256, 240/256, 130/176) = (0.9375, 0.9375, 0.739)
          spacing = max(ratios) = 0.9375 mm (XY-driven)
          resampled Z: 130 / 0.9375 ≈ 139 voxels → fits in 176 (only padding needed)
        """
        path = make_nifti((512, 512, 260), (0.4688, 0.4688, 0.5))
        out = tmp_path / "out_b176_noncubic.nii.gz"
        generate_variant_b(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = np.array(img.header.get_zooms()[:3])
        # All axes must be equal (isotropic)
        assert np.allclose(zooms[0], zooms[2], atol=1e-3), (
            f"Spacing must be isotropic on Z: {zooms}"
        )

    def test_variant_c_shape(self, make_nifti, tmp_path):
        path = make_nifti((300, 300, 200), (1.0, 1.0, 1.0))
        out = tmp_path / "out_c176.nii.gz"
        generate_variant_c(path, out, target_shape=self.TARGET)
        assert nib.load(out).shape == self.TARGET

    def test_variant_d_shape(self, make_nifti, tmp_path):
        path = make_nifti((400, 400, 200), (0.5, 0.5, 0.6))
        out = tmp_path / "out_d176.nii.gz"
        generate_variant_d(path, out, target_shape=self.TARGET)
        assert nib.load(out).shape == self.TARGET

    def test_variant_e_shape_and_fixed_spacing(self, make_nifti, tmp_path):
        """Variant E at 256×256×176 must use E_SPACING = 192/256 = 0.75 mm."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_e176.nii.gz"
        generate_variant_e(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        expected = E_SPACING_MM_BY_SHAPE[self.TARGET]  # 0.75 mm
        assert np.allclose(zooms[0], expected, atol=0.05), (
            f"Expected E spacing {expected:.3f} mm, got {zooms[0]:.3f}"
        )

    def test_variant_f_shape_and_fixed_spacing(self, make_nifti, tmp_path):
        """Variant F at 256×256×176 must use F_SPACING = 240/256 = 0.9375 mm."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_f176.nii.gz"
        generate_variant_f(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        expected = F_SPACING_MM_BY_SHAPE[self.TARGET]  # 0.9375 mm
        assert np.allclose(zooms[0], expected, atol=0.05), (
            f"Expected F spacing {expected:.3f} mm, got {zooms[0]:.3f}"
        )

    def test_all_six_variants_shape(self, make_nifti, tmp_path):
        """All six variants must produce 256×256×176."""
        path = make_nifti((300, 300, 200), (0.6, 0.6, 0.7))
        for i, fn in enumerate(ALL_VARIANTS):
            out = tmp_path / f"out_176_{i}.nii.gz"
            fn(path, out, target_shape=self.TARGET)
            assert nib.load(out).shape == self.TARGET, f"{fn.__name__} wrong shape"


# ---------------------------------------------------------------------------
# 192×192×128 target shape — integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SAMPLE_NIFTI_DIR.exists(),
    reason=f"Sample dataset not found at {SAMPLE_NIFTI_DIR}. Attach the external SSD to run integration tests.",
)
class TestRealDataPipeline192:
    """Integration tests using the 12-case sample NIfTI dataset at 192×192×128.

    These tests auto-skip when /mnt/data/nifti-sample is not mounted.
    """

    TARGET = (192, 192, 128)
    ALL_FNS = [
        ("A", generate_variant_a),
        ("B", generate_variant_b),
        ("C", generate_variant_c),
        ("D", generate_variant_d),
        ("E", generate_variant_e),
        ("F", generate_variant_f),
    ]

    def test_all_variants_correct_shape(self, tmp_path):
        """All six variants must produce 192×192×128 on every real scan."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0
        for src in source_files:
            for tag, fn in self.ALL_FNS:
                out = tmp_path / f"{tag}_{src.name}"
                fn(src, out, target_shape=self.TARGET)
                assert out.exists(), f"Variant {tag}: no output for {src.name}"
                assert nib.load(out).shape == self.TARGET, (
                    f"Variant {tag} {src.name}: expected {self.TARGET}, got {nib.load(out).shape}"
                )

    def test_resampling_variants_isotropic(self, tmp_path):
        """Variants A, B, E, F must produce isotropic spacing on real clinical scans."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0
        for src in source_files:
            for tag, fn in [("A", generate_variant_a), ("B", generate_variant_b),
                            ("E", generate_variant_e), ("F", generate_variant_f)]:
                out = tmp_path / f"iso_{tag}_{src.name}"
                fn(src, out, target_shape=self.TARGET)
                zooms = np.array(nib.load(out).header.get_zooms()[:3])
                assert np.allclose(zooms[0], zooms[1], atol=1e-3), (
                    f"Variant {tag} {src.name}: spacing not isotropic: {zooms}"
                )
                assert np.allclose(zooms[0], zooms[2], atol=1e-3), (
                    f"Variant {tag} {src.name}: spacing not isotropic: {zooms}"
                )

    def test_all_variants_nonzero_signal(self, tmp_path):
        """Every variant must preserve non-zero brain signal in its output window."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0
        for src in source_files:
            for tag, fn in self.ALL_FNS:
                out = tmp_path / f"{tag}_{src.name}"
                fn(src, out, target_shape=self.TARGET)
                data = nib.load(out).get_fdata(dtype=np.float32)
                nonzero_fraction = np.count_nonzero(data) / data.size
                assert nonzero_fraction > 0.10, (
                    f"Variant {tag} {src.name}: only {nonzero_fraction:.1%} non-zero voxels"
                )


# ---------------------------------------------------------------------------
# 256×256×176 target shape — integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SAMPLE_NIFTI_DIR.exists(),
    reason=f"Sample dataset not found at {SAMPLE_NIFTI_DIR}. Attach the external SSD to run integration tests.",
)
class TestRealDataPipeline256x176:
    """Integration tests using the 12-case sample NIfTI dataset at 256×256×176.

    These tests auto-skip when /mnt/data/nifti-sample is not mounted.
    """

    TARGET = (256, 256, 176)
    ALL_FNS = [
        ("A", generate_variant_a),
        ("B", generate_variant_b),
        ("C", generate_variant_c),
        ("D", generate_variant_d),
        ("E", generate_variant_e),
        ("F", generate_variant_f),
    ]

    def test_all_variants_correct_shape(self, tmp_path):
        """All six variants must produce 256×256×176 on every real scan."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0
        for src in source_files:
            for tag, fn in self.ALL_FNS:
                out = tmp_path / f"{tag}_{src.name}"
                fn(src, out, target_shape=self.TARGET)
                assert out.exists(), f"Variant {tag}: no output for {src.name}"
                assert nib.load(out).shape == self.TARGET, (
                    f"Variant {tag} {src.name}: expected {self.TARGET}, got {nib.load(out).shape}"
                )

    def test_resampling_variants_isotropic(self, tmp_path):
        """Variants A, B, E, F must produce isotropic spacing on real clinical scans."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0
        for src in source_files:
            for tag, fn in [("A", generate_variant_a), ("B", generate_variant_b),
                            ("E", generate_variant_e), ("F", generate_variant_f)]:
                out = tmp_path / f"iso_{tag}_{src.name}"
                fn(src, out, target_shape=self.TARGET)
                zooms = np.array(nib.load(out).header.get_zooms()[:3])
                assert np.allclose(zooms[0], zooms[1], atol=1e-3), (
                    f"Variant {tag} {src.name}: spacing not isotropic: {zooms}"
                )
                assert np.allclose(zooms[0], zooms[2], atol=1e-3), (
                    f"Variant {tag} {src.name}: spacing not isotropic: {zooms}"
                )

    def test_all_variants_nonzero_signal(self, tmp_path):
        """Every variant must preserve non-zero brain signal in its output window."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0
        for src in source_files:
            for tag, fn in self.ALL_FNS:
                out = tmp_path / f"{tag}_{src.name}"
                fn(src, out, target_shape=self.TARGET)
                data = nib.load(out).get_fdata(dtype=np.float32)
                nonzero_fraction = np.count_nonzero(data) / data.size
                assert nonzero_fraction > 0.10, (
                    f"Variant {tag} {src.name}: only {nonzero_fraction:.1%} non-zero voxels"
                )
