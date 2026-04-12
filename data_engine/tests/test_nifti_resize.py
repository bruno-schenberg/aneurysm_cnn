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
)

# Integration tests use real NIfTI files from this path. Tests auto-skip when
# the path is not mounted so the unit test suite always runs without the SSD.
SAMPLE_NIFTI_DIR = Path("/mnt/data/nifti-sample")

ALL_VARIANTS = [
    generate_variant_a,
    generate_variant_b,
    generate_variant_c,
    generate_variant_d,
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
    """Variant B output must be 128³ with isotropic spacing (not necessarily 1 mm).

    Variant B computes target_spacing = max_physical_extent / 128, so spacing
    depends on the input volume. For (200, 200, 150) at (0.8, 0.8, 1.2):
      max_physical_extent = max(160, 160, 180) = 180 mm
      target_spacing = 180 / 128 ≈ 1.406 mm
    The test checks isotropy (all three axes equal), not a fixed value.
    """
    path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
    out = tmp_path / "out_b.nii.gz"
    generate_variant_b(path, out)
    img = nib.load(out)
    assert img.shape == TARGET_SHAPE
    zooms = img.header.get_zooms()[:3]
    assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
    assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"


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


def test_all_variants_no_exception_on_valid_input(make_nifti, tmp_path):
    """All four variants complete without raising on a standard input."""
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
    """Variants A and B must have header zooms consistent with affine diagonal (FR-013).

    Variant A always produces 1 mm isotropic spacing.
    Variant B produces max_extent/128 isotropic spacing (not necessarily 1 mm).
    Both must have header zooms matching affine diagonal zooms.
    """
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

    # Variant A specifically must produce 1 mm isotropic spacing
    out_a = tmp_path / "out_a_spacing.nii.gz"
    generate_variant_a(path, out_a)
    zooms_a = np.array(nib.load(out_a).header.get_zooms()[:3])
    assert np.allclose(zooms_a, 1.0, atol=1e-3), f"Variant A expected 1 mm, got {zooms_a}"


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
# 256×256×128 target shape — unit tests (T016)
# ---------------------------------------------------------------------------


class TestVariant256Shape:
    """Unit tests for all four variants at the 256×256×128 target shape.

    All tests use synthetic NIfTI volumes and require no external data.
    """

    TARGET = (256, 256, 128)

    def test_variant_a_shape_and_spacing(self, make_nifti, tmp_path):
        """Variant A at 256×256×128 must output that shape at 0.9375 mm isotropic."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_a256.nii.gz"
        generate_variant_a(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        assert np.allclose(zooms, 0.9375, atol=1e-3), (
            f"Expected 0.9375 mm isotropic, got {zooms}"
        )

    def test_variant_b_shape_and_spacing(self, make_nifti, tmp_path):
        """Variant B at 256×256×128 must output that shape with isotropic spacing."""
        path = make_nifti((200, 200, 150), (0.8, 0.8, 1.2))
        out = tmp_path / "out_b256.nii.gz"
        generate_variant_b(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = img.header.get_zooms()[:3]
        assert np.allclose(zooms[0], zooms[1], atol=1e-3), "Spacing must be isotropic"
        assert np.allclose(zooms[0], zooms[2], atol=1e-3), "Spacing must be isotropic"

    def test_variant_c_shape(self, make_nifti, tmp_path):
        """Variant C at 256×256×128 must output that shape with no interpolation."""
        path = make_nifti((300, 300, 150), (1.0, 1.0, 1.0))
        out = tmp_path / "out_c256.nii.gz"
        generate_variant_c(path, out, target_shape=self.TARGET)
        assert nib.load(out).shape == self.TARGET

    def test_variant_d_shape(self, make_nifti, tmp_path):
        """Variant D at 256×256×128 must output that shape via trilinear resize."""
        path = make_nifti((400, 400, 200), (0.5, 0.5, 0.6))
        out = tmp_path / "out_d256.nii.gz"
        generate_variant_d(path, out, target_shape=self.TARGET)
        assert nib.load(out).shape == self.TARGET

    def test_variant_b_isotropy_clinical_geometry(self, make_nifti, tmp_path):
        """Variant B at 256×256×128 applied to clinical scan geometry must produce ~0.938 mm isotropic spacing.

        Clinical geometry: 512×512 XY at 0.4688 mm/pixel, 208 slices at 0.6 mm/slice.
        Expected: max_physical_extent = max(512*0.4688, 512*0.4688, 208*0.6)
                                       = max(240.0, 240.0, 124.8) = 240.0 mm
                  target_spacing = 240.0 / 256 = 0.9375 mm
        """
        path = make_nifti((512, 512, 208), (0.4688, 0.4688, 0.6))
        out = tmp_path / "out_b256_clinical.nii.gz"
        generate_variant_b(path, out, target_shape=self.TARGET)
        img = nib.load(out)
        assert img.shape == self.TARGET
        zooms = np.array(img.header.get_zooms()[:3])
        expected = 0.938
        tolerance = expected * 0.05  # ±5%
        assert np.all(np.abs(zooms - expected) <= tolerance), (
            f"Expected spacing ≈{expected} mm ±5%, got {zooms}"
        )


# ---------------------------------------------------------------------------
# 256×256×128 target shape — integration tests (T017)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SAMPLE_NIFTI_DIR.exists(),
    reason=f"Sample dataset not found at {SAMPLE_NIFTI_DIR}. Attach the external SSD to run integration tests.",
)
class TestRealDataPipeline256:
    """Integration tests using the 12-case sample NIfTI dataset.

    These tests auto-skip when /mnt/data/nifti-sample is not mounted so the
    unit test suite always passes without the external SSD attached.
    """

    def test_variant_b_all_files_produce_correct_shape(self, tmp_path):
        """All 12 real NIfTI source files must produce (256, 256, 128) output with variant B."""
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0, (
            f"No .nii.gz files found under {SAMPLE_NIFTI_DIR}; check sample dataset structure."
        )

        target = (256, 256, 128)
        for src in source_files:
            out = tmp_path / src.name
            generate_variant_b(src, out, target_shape=target)
            assert out.exists(), f"No output produced for {src.name}"
            assert nib.load(out).shape == target, (
                f"{src.name}: expected shape {target}, got {nib.load(out).shape}"
            )

    def test_variant_b_spacing_is_isotropic(self, tmp_path):
        """Variant B at 256×256×128 on real clinical scans must produce isotropic spacing.

        Variant B computes target_spacing = max_physical_extent / 256, which varies
        per scan depending on actual field-of-view. The invariant that must always hold
        is that the resulting spacing is isotropic (all three axes equal within ±1%),
        confirming that the single-pass resampling preserved proportions correctly.
        """
        source_files = sorted(SAMPLE_NIFTI_DIR.rglob("*.nii.gz"))
        assert len(source_files) > 0

        target = (256, 256, 128)

        for src in source_files:
            out = tmp_path / f"iso_{src.name}"
            generate_variant_b(src, out, target_shape=target)
            zooms = np.array(nib.load(out).header.get_zooms()[:3])
            assert np.allclose(zooms[0], zooms[1], atol=1e-3), (
                f"{src.name}: spacing not isotropic: {zooms}"
            )
            assert np.allclose(zooms[0], zooms[2], atol=1e-3), (
                f"{src.name}: spacing not isotropic: {zooms}"
            )
