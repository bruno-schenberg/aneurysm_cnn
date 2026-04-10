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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_engine.src.nifti_resize import (
    _load,
    _variant_a_from_data,
    _variant_b_from_data,
    _variant_c_from_data,
    _variant_d_from_data,
)

INPUT_NIFTI_PATH = Path("/mnt/data/cases-3/nifti")

OUTPUT_PATHS = {
    "A": Path("/mnt/data/cases-3/dataset_A_resampled_cropped"),
    "B": Path("/mnt/data/cases-3/dataset_B_resampled_shrunk"),
    "C": Path("/mnt/data/cases-3/dataset_C_cropped"),
    "D": Path("/mnt/data/cases-3/dataset_D_shrunk"),
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
# ---------------------------------------------------------------------------

def _task_a(args: tuple[Path, Path]) -> tuple[str, str | None]:
    file_path, output_path = args
    label = f"{file_path.name}:A"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_a_from_data(data, affine, output_path)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_b(args: tuple[Path, Path]) -> tuple[str, str | None]:
    file_path, output_path = args
    label = f"{file_path.name}:B"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_b_from_data(data, affine, output_path)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_c(args: tuple[Path, Path]) -> tuple[str, str | None]:
    file_path, output_path = args
    label = f"{file_path.name}:C"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_c_from_data(data, affine, output_path)
        del data, affine
        gc.collect()
        return label, None
    except Exception as exc:
        return label, f"{type(exc).__name__}: {exc}"


def _task_d(args: tuple[Path, Path]) -> tuple[str, str | None]:
    file_path, output_path = args
    label = f"{file_path.name}:D"
    if output_path.exists():
        return label, None
    try:
        data, affine = _load(file_path)
        _variant_d_from_data(data, affine, output_path)
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
    """
    task_fn = _TASK_FN[variant_key]
    out_base = OUTPUT_PATHS[variant_key]

    tasks: list[tuple[Path, Path]] = []
    for file_path in nifti_files:
        class_dir = file_path.parent.name
        out_dir = out_base / class_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((file_path, out_dir / file_path.name))

    total = len(tasks)
    failures: list[tuple[str, str]] = []
    done = 0

    logging.info("=== Variant %s | %d files | %d workers ===", variant_key, total, n_workers)

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
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            f"Parallel workers per variant (default: {DEFAULT_WORKERS}). "
            "Variants A/B (MONAI Spacing) peak at ~4 GB per worker; "
            "reduce to 2 if you see OOM errors on large scans."
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

    nifti_files = list(INPUT_NIFTI_PATH.rglob("*/*.nii.gz"))
    if not nifti_files:
        logging.warning("No files in subdirectories; checking top level.")
        nifti_files = list(INPUT_NIFTI_PATH.glob("*.nii.gz"))
    if not nifti_files:
        logging.error("No .nii.gz files found under %s", INPUT_NIFTI_PATH)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(nifti_files)

    n_workers = min(args.workers, len(nifti_files))
    logging.info(
        "%d files | variants: %s | %d workers per variant",
        len(nifti_files),
        " ".join(args.variants),
        n_workers,
    )

    all_failures: list[tuple[str, str]] = []
    for variant_key in args.variants:
        failures = _run_variant(variant_key, nifti_files, n_workers)
        all_failures.extend(failures)

    if all_failures:
        logging.error("%d total task(s) failed across all variants:", len(all_failures))
        for label, error in all_failures:
            logging.error("  %-45s %s", label, error)
        sys.exit(1)

    logging.info("All variants complete.")


if __name__ == "__main__":
    main()
