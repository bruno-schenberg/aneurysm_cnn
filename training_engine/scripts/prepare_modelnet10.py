"""
Convert ModelNet10 OFF meshes to 128³ NIfTI volumes for pipeline validation.

Reads .off mesh files for two chosen classes, voxelizes each mesh into a
128×128×128 binary occupancy grid, and saves as .nii.gz in the folder
layout expected by training_engine/src/data.py:

    <output_dir>/0/<filename>.nii.gz   (class 0 = bathtub)
    <output_dir>/1/<filename>.nii.gz   (class 1 = chair)

Usage:
    cd training_engine
    source .venv/bin/activate
    python scripts/prepare_modelnet10.py --modelnet10-root <path> --output-dir <path>

The voxelization pipeline follows the same approach as the reference notebook
(intro-3d-cv/resnet3d_classification_short.ipynb): surface point sampling
with area-weighted barycentric coordinates, then quantization to a voxel grid.
"""

import argparse
import random
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────

VOXEL_RESOLUTION = 128
NUM_POINTS = 20000  # more points than the 32³ notebook to fill 128³ well
CLASSES = {"bathtub": 0, "chair": 1}


# ── OFF file reading ──────────────────────────────────────────────────────

def read_off(file_path):
    """Parse an OFF mesh file and return (vertices, faces)."""
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        if first_line == "OFF":
            counts_line = f.readline().strip()
        elif first_line.startswith("OFF"):
            counts_line = first_line[3:].strip()
        else:
            raise ValueError(f"Not a valid OFF file: {file_path}")

        num_verts, num_faces, _ = (int(s) for s in counts_line.split())

        vertices = np.array(
            [[float(c) for c in f.readline().strip().split()] for _ in range(num_verts)],
            dtype=np.float32,
        )
        faces = [
            [int(idx) for idx in f.readline().strip().split()][1:]
            for _ in range(num_faces)
        ]
    return vertices, faces


# ── Point sampling ────────────────────────────────────────────────────────

def _triangle_areas(vertices, faces):
    """Compute area of each triangular face."""
    v0 = vertices[[f[0] for f in faces]]
    v1 = vertices[[f[1] for f in faces]]
    v2 = vertices[[f[2] for f in faces]]
    crosses = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(crosses, axis=1)


def sample_surface_points(vertices, faces, num_points):
    """Sample points uniformly from mesh surface using area-weighted barycentric sampling."""
    areas = _triangle_areas(vertices, faces)
    if areas.sum() == 0:
        warnings.warn("Mesh has zero surface area, returning zeros.")
        return np.zeros((num_points, 3), dtype=np.float32)

    probs = areas / areas.sum()
    face_indices = np.random.choice(len(faces), size=num_points, p=probs)

    faces_arr = np.array(faces)
    sampled = faces_arr[face_indices]
    v0 = vertices[sampled[:, 0]]
    v1 = vertices[sampled[:, 1]]
    v2 = vertices[sampled[:, 2]]

    r1 = np.random.rand(num_points, 1).astype(np.float32)
    r2 = np.random.rand(num_points, 1).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    points = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
    return points


# ── Voxelization ──────────────────────────────────────────────────────────

def voxelize(points, resolution):
    """Convert (N, 3) point cloud to a (resolution³) binary occupancy grid."""
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    if points.shape[0] == 0:
        return grid

    shifted = points - points.min(axis=0)
    max_extent = shifted.max()
    if max_extent == 0:
        return grid

    normalized = shifted / max_extent
    indices = (normalized * (resolution - 1)).astype(np.int32)
    indices = np.clip(indices, 0, resolution - 1)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return grid


# ── Conversion ────────────────────────────────────────────────────────────

def convert_class(modelnet10_root, output_dir, class_name, class_label):
    """Convert all OFF files for one class to NIfTI volumes."""
    class_dir = modelnet10_root / class_name
    out_dir = output_dir / str(class_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    off_files = []
    for split in ("train", "test"):
        split_dir = class_dir / split
        if split_dir.is_dir():
            off_files.extend(sorted(split_dir.glob("*.off")))

    print(f"  {class_name} → class {class_label}: {len(off_files)} meshes")
    skipped = 0

    for off_path in off_files:
        out_path = out_dir / f"{off_path.stem}.nii.gz"
        if out_path.exists():
            continue

        try:
            vertices, faces = read_off(off_path)
            points = sample_surface_points(vertices, faces, NUM_POINTS)
            grid = voxelize(points, VOXEL_RESOLUTION)

            # Save as NIfTI with identity affine (no real-world coordinates needed)
            img = nib.Nifti1Image(grid, affine=np.eye(4))
            nib.save(img, str(out_path))
        except Exception as e:
            print(f"    SKIP {off_path.name}: {e}")
            skipped += 1

    converted = len(off_files) - skipped
    print(f"    {converted} converted, {skipped} skipped")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert ModelNet10 meshes to 128³ NIfTI volumes")
    parser.add_argument(
        "--modelnet10-root",
        type=Path,
        required=True,
        help="Path to extracted ModelNet10 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/datasets/modelnet10_pair"),
        help="Output directory for NIfTI files",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.modelnet10_root.is_dir():
        parser.error(f"ModelNet10 root not found: {args.modelnet10_root}")

    print(f"ModelNet10 root: {args.modelnet10_root}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Resolution:      {VOXEL_RESOLUTION}³")
    print(f"Surface points:  {NUM_POINTS}")
    print()

    total = 0
    for class_name, class_label in CLASSES.items():
        total += convert_class(args.modelnet10_root, args.output_dir, class_name, class_label)

    print(f"\nDone. {total} volumes written to {args.output_dir}")


if __name__ == "__main__":
    main()
