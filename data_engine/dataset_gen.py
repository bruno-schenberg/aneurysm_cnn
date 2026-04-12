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
    VALID_TARGET_SHAPES,
)

# Default source directory. Overridden at runtime by --input-dir.
# Kept as a named constant so the intent is clear if someone reads the source
# without looking at the argparse section.
DEFAULT_INPUT_DIR = Path("/mnt/data/cases-3/nifti")

# Output directories keyed first by resolution string, then by variant key.
#
# The 128x128x128 paths are identical to the pre-feature paths — callers that
# do not pass --target-shape continue to write to the same locations.
#
# To add a new resolution: add an entry to VALID_TARGET_SHAPES in
# nifti_resize.py AND a matching entry here. No other code changes are needed.
OUTPUT_PATHS: Dict[str, Dict[str, Path]] = {
    "128x128x128": {
        "A": Path("/mnt/data/cases-3/dataset_A_resampled_cropped"),
        "B": Path("/mnt/data/cases-3/dataset_B_resampled_shrunk"),
        "C": Path("/mnt/data/cases-3/dataset_C_cropped"),
        "D": Path("/mnt/data/cases-3/dataset_D_shrunk"),
    },
    "256x256x128": {
        "A": Path("/mnt/data/cases-3/dataset_A256_resampled_cropped"),
        "B": Path("/mnt/data/cases-3/dataset_B256_resampled_shrunk"),
        "C": Path("/mnt/data/cases-3/dataset_C256_cropped"),
        "D": Path("/mnt/data/cases-3/dataset_D256_shrunk"),
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


_TASK_FN = {"A": _task_a, "B": _task_b, "C": _task_c, "D": _task_d}


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
        variant_key: One of 'A', 'B', 'C', 'D'.
        nifti_files: Flat list of source ``.nii.gz`` paths (may include class
            subdirectory structure which is mirrored in the output).
        n_workers: Number of parallel worker processes.
        target_shape: Output voxel grid passed through to the variant function.
        target_shape_key: String key into ``OUTPUT_PATHS`` (e.g. '128x128x128').
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
            "Generate preprocessing variants (A–D) of every NIfTI file. "
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
        default="128x128x128",
        metavar="SHAPE",
        help=(
            "Output voxel grid for all variants. "
            f"Supported values: {', '.join(VALID_TARGET_SHAPES)}. "
            "Default: 128x128x128 (existing behaviour). "
            "Each shape writes to a separate output directory set so the "
            "128x128x128 and 256x256x128 datasets can coexist on disk."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            f"Parallel workers per variant (default: {DEFAULT_WORKERS}). "
            "Variants A/B (MONAI Spacing) peak at ~4 GB per worker at 128³; "
            "256x256x128 volumes are 8× larger — reduce to 2 if you see OOM errors."
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
        choices=["A", "B", "C", "D"],
        default=["C", "D", "A", "B"],
        metavar="V",
        help=(
            "Variants to generate, in order (default: C D A B). "
            "C and D run first as they are cheaper and confirm the pipeline "
            "works before the more expensive A/B resampling runs."
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
