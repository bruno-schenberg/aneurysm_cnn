import os

# Must be set before ANY library import. scipy, numpy, and MONAI read these
# variables when their C extensions are first loaded. Setting them after import
# has no effect on already-initialised OpenMP/BLAS thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gc
import logging
import multiprocessing
import random
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_engine.src.nifti_resize import (
    _load,
    _variant_a_from_data,
    _variant_b_from_data,
    _variant_c_from_data,
    _variant_d_from_data,
    _variant_e_from_data,
    _variant_f_from_data,
    VALID_TARGET_SHAPES,
)

# Default source directory. Overridden at runtime by --input-dir.
DEFAULT_INPUT_DIR = Path("/mnt/data/cases-3/nifti")

# Output directories keyed first by resolution string, then by variant key.
#
# The 128x128x128 paths are identical to the pre-redesign paths — callers that
# do not pass --target-shape continue to write to the same locations.
#
# Naming convention:
#   - 192x192x128 datasets use suffix "192"  (e.g. dataset_A192)
#   - 256x256x176 datasets use suffix "176"  (e.g. dataset_A176)
#
# To add a new resolution: add an entry to VALID_TARGET_SHAPES in
# nifti_resize.py AND a matching entry here. No other code changes are needed.
OUTPUT_PATHS: Dict[str, Dict[str, Path]] = {
    "128x128x128": {
        "A": Path("/mnt/data/cases-3/dataset_A_resampled_cropped"),
        "B": Path("/mnt/data/cases-3/dataset_B_resampled_shrunk"),
        "C": Path("/mnt/data/cases-3/dataset_C_cropped"),
        "D": Path("/mnt/data/cases-3/dataset_D_shrunk"),
        "E": Path("/mnt/data/cases-3/dataset_E_fixed_cropped"),
        "F": Path("/mnt/data/cases-3/dataset_F_fixed_shrunk"),
    },
    "192x192x128": {
        "A": Path("/mnt/data/cases-3/dataset_A192"),
        "B": Path("/mnt/data/cases-3/dataset_B192"),
        "C": Path("/mnt/data/cases-3/dataset_C192"),
        "D": Path("/mnt/data/cases-3/dataset_D192"),
        "E": Path("/mnt/data/cases-3/dataset_E192"),
        "F": Path("/mnt/data/cases-3/dataset_F192"),
    },
    "256x256x176": {
        "A": Path("/mnt/data/cases-3/dataset_A176"),
        "B": Path("/mnt/data/cases-3/dataset_B176"),
        "C": Path("/mnt/data/cases-3/dataset_C176"),
        "D": Path("/mnt/data/cases-3/dataset_D176"),
        "E": Path("/mnt/data/cases-3/dataset_E176"),
        "F": Path("/mnt/data/cases-3/dataset_F176"),
    },
}

DEFAULT_WORKERS = 4


def _worker_init() -> None:
    import torch
    torch.set_num_threads(1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# One top-level worker function per variant.
#
# These are defined at module level (not inside a dict or closure) so they are
# unambiguously importable by name in forked workers. They call the variant
# function directly — no dictionary indirection, no indirect references.
#
# Each args tuple carries (file_path, output_path, target_shape). The
# target_shape is a plain tuple so it is fully picklable across process
# boundaries. Using closures or functools.partial is unsafe here because MONAI
# may set the start method to 'forkserver', which spawns workers from a clean
# server process that cannot inherit closure state.
# ---------------------------------------------------------------------------

def _task_a(args: tuple[Path, Path, tuple[int, int, int]]) -> tuple[str, str | None]:
    file_path, output_path, target_shape = args
    label = f"{file_path.name}:A"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_a_from_data(data, affine, output_path, target_shape=target_shape)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_b(args: tuple[Path, Path, tuple[int, int, int]]) -> tuple[str, str | None]:
    file_path, output_path, target_shape = args
    label = f"{file_path.name}:B"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_b_from_data(data, affine, output_path, target_shape=target_shape)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_c(args: tuple[Path, Path, tuple[int, int, int]]) -> tuple[str, str | None]:
    file_path, output_path, target_shape = args
    label = f"{file_path.name}:C"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_c_from_data(data, affine, output_path, target_shape=target_shape)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_d(args: tuple[Path, Path, tuple[int, int, int]]) -> tuple[str, str | None]:
    file_path, output_path, target_shape = args
    label = f"{file_path.name}:D"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_d_from_data(data, affine, output_path, target_shape=target_shape)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_e(args: tuple[Path, Path, tuple[int, int, int]]) -> tuple[str, str | None]:
    file_path, output_path, target_shape = args
    label = f"{file_path.name}:E"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_e_from_data(data, affine, output_path, target_shape=target_shape)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_f(args: tuple[Path, Path, tuple[int, int, int]]) -> tuple[str, str | None]:
    file_path, output_path, target_shape = args
    label = f"{file_path.name}:F"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_f_from_data(data, affine, output_path, target_shape=target_shape)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


_TASK_FN = {
    "A": _task_a,
    "B": _task_b,
    "C": _task_c,
    "D": _task_d,
    "E": _task_e,
    "F": _task_f,
}


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------

def _run_variant(
    variant_key: str,
    nifti_files: list[Path],
    n_workers: int,
    target_shape: tuple[int, int, int],
    target_shape_key: str,
) -> list[tuple[str, str]]:
    """
    Run one variant across all files using a dedicated fork pool.

    A fresh pool is created for each variant so that state from a previous
    variant's workers cannot leak into the next. maxtasksperchild=1 ensures
    the OS reclaims each worker's memory immediately after its one task.

    We explicitly request the 'fork' start method. Without this, PyTorch or
    MONAI may have already called set_start_method('forkserver'), which spawns
    workers from a clean server process that does not inherit sys.path. Under
    forkserver the nifti_resize imports silently fail, tasks never return a
    result, and the pool stalls.

    Args:
        variant_key: One of 'A', 'B', 'C', 'D', 'E', 'F'.
        nifti_files: Flat list of source ``.nii.gz`` paths (may include class
            subdirectory structure which is mirrored in the output).
        n_workers: Number of parallel worker processes.
        target_shape: Output voxel grid passed through to the variant function.
        target_shape_key: String key into ``OUTPUT_PATHS`` (e.g. '192x192x128').
            Determines which output directory set is used.
    """
    task_fn = _TASK_FN[variant_key]
    out_base = OUTPUT_PATHS[target_shape_key][variant_key]

    tasks: list[tuple[Path, Path, tuple[int, int, int]]] = []
    for file_path in nifti_files:
        class_dir = file_path.parent.name
        out_dir = out_base / class_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((file_path, out_dir / file_path.name, target_shape))

    total = len(tasks)
    failures: list[tuple[str, str]] = []
    done = 0

    logging.info(
        "=== Variant %s | shape %s | %d files | %d workers ===",
        variant_key, target_shape_key, total, n_workers,
    )

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_worker_init, maxtasksperchild=1) as pool:
        for label, error in pool.imap_unordered(task_fn, tasks):
            done += 1
            if error:
                failures.append((label, error))
                logging.warning("[%d/%d] FAIL  %s — %s", done, total, label, error)
            else:
                logging.info("[%d/%d] ok    %s", done, total, label)

    if failures:
        logging.error("Variant %s: %d/%d file(s) failed.", variant_key, len(failures), total)
    else:
        logging.info("Variant %s: all %d files done.", variant_key, total)

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate preprocessing variants (A–F) of every NIfTI file. "
            "Each variant is processed in its own pool loop. "
            "Each worker handles one file and exits (maxtasksperchild=1)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        metavar="DIR",
        help=(
            f"Root folder containing .nii.gz source files, optionally organised "
            f"into class subdirectories (0/ and 1/). "
            f"Default: {DEFAULT_INPUT_DIR}. "
            "Override with the HPC cluster path (e.g. $HOME/data/cases-3/nifti)."
        ),
    )
    parser.add_argument(
        "--target-shape",
        choices=list(VALID_TARGET_SHAPES),
        default="192x192x128",
        metavar="SHAPE",
        help=(
            "Output voxel grid for all variants. "
            f"Supported values: {', '.join(VALID_TARGET_SHAPES)}. "
            "Default: 192x192x128. "
            "Each shape writes to a separate output directory set."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            f"Parallel workers per variant (default: {DEFAULT_WORKERS}). "
            "Variants A/B/E/F (MONAI Spacing) peak at ~4 GB per worker at 192³; "
            "256x256x176 volumes are larger — reduce to 2 if you see OOM errors."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for the file list (default: 42).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["A", "B", "C", "D", "E", "F"],
        default=["C", "D", "A", "B", "E", "F"],
        metavar="V",
        help=(
            "Variants to generate, in order (default: C D A B E F). "
            "C and D run first as they are cheaper and confirm the pipeline "
            "works before the more expensive resampling runs."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        logging.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    target_shape_key: str = args.target_shape
    target_shape: tuple[int, int, int] = VALID_TARGET_SHAPES[target_shape_key]

    nifti_files = list(input_dir.rglob("*/*.nii.gz"))
    if not nifti_files:
        logging.warning("No files in subdirectories; checking top level.")
        nifti_files = list(input_dir.glob("*.nii.gz"))
    if not nifti_files:
        logging.error("No .nii.gz files found under %s", input_dir)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(nifti_files)

    n_workers = min(args.workers, len(nifti_files))
    logging.info(
        "%d files | shape: %s | variants: %s | %d workers per variant",
        len(nifti_files),
        target_shape_key,
        " ".join(args.variants),
        n_workers,
    )

    all_failures: list[tuple[str, str]] = []
    for variant_key in args.variants:
        failures = _run_variant(
            variant_key, nifti_files, n_workers, target_shape, target_shape_key
        )
        all_failures.extend(failures)

    if all_failures:
        logging.error("%d total task(s) failed across all variants:", len(all_failures))
        for label, error in all_failures:
            logging.error("  %-45s %s", label, error)
        sys.exit(1)

    logging.info("All variants complete.")


if __name__ == "__main__":
    main()
