import argparse
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_engine.src.nifti_resize import (
    generate_variant_a,
    generate_variant_b,
    generate_variant_c,
    generate_variant_d,
    generate_variant_e,
)

INPUT_NIFTI_PATH = Path("/mnt/data/cases-3/nifti")

# One output directory per variant, keyed by variant letter
OUTPUT_PATHS = {
    "A": Path("/mnt/data/cases-3/dataset_A_resampled_cropped"),
    "B": Path("/mnt/data/cases-3/dataset_B_resampled_shrunk"),
    "C": Path("/mnt/data/cases-3/dataset_C_cropped"),
    "D": Path("/mnt/data/cases-3/dataset_D_shrunk"),
    "E": Path("/mnt/data/cases-3/dataset_E_isotropic_padded"),
}

DEFAULT_WORKERS = 3

# Workers are recycled after this many files to flush MONAI/torch allocator
# caches that accumulate across tasks.  multiprocessing.Pool handles recycling
# without the simultaneous-recycle deadlock present in ProcessPoolExecutor.
MAX_TASKS_PER_WORKER = 10


def _worker_logging_init() -> None:
    """Configure a StreamHandler in each worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )


def process_file(file_path: Path, output_dirs: dict[str, Path]) -> str:
    """Generate all five dataset variants for a single NIfTI file."""
    filename = file_path.name
    for variant, fn, key in [
        ("A", generate_variant_a, "A"),
        ("B", generate_variant_b, "B"),
        ("C", generate_variant_c, "C"),
        ("D", generate_variant_d, "D"),
        ("E", generate_variant_e, "E"),
    ]:
        logging.info("%s → variant %s", filename, variant)
        fn(file_path, output_dirs[key] / filename)
    return f"Done: {filename}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate five preprocessing variants of every NIfTI file."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Input: %s", INPUT_NIFTI_PATH)
    for key, path in OUTPUT_PATHS.items():
        logging.info("Variant %s → %s", key, path)

    # Discover NIfTI files; they live in class subdirectories (0/ and 1/)
    nifti_files = list(INPUT_NIFTI_PATH.rglob("*/*.nii.gz"))
    if not nifti_files:
        logging.warning("No .nii.gz files found in subdirectories; checking top level.")
        nifti_files = list(INPUT_NIFTI_PATH.glob("*.nii.gz"))
    if not nifti_files:
        logging.error("No .nii.gz files found. Exiting.")
        return

    logging.info("Found %d files to process with %d workers.", len(nifti_files), args.workers)

    # Build per-file output directories keyed by variant letter, mirroring the
    # class subdirectory structure (0/ and 1/) from the input tree.
    file_output_dirs: list[tuple[Path, dict[str, Path]]] = []
    for file_path in nifti_files:
        class_dir = file_path.parent.name
        dirs = {key: OUTPUT_PATHS[key] / class_dir for key in OUTPUT_PATHS}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        file_output_dirs.append((file_path, dirs))

    with Pool(
        processes=args.workers,
        initializer=_worker_logging_init,
        maxtasksperchild=MAX_TASKS_PER_WORKER,
    ) as pool:
        for fp, dirs in file_output_dirs:
            pool.apply_async(
                process_file,
                args=(fp, dirs),
                callback=lambda msg: logging.info(msg),
                error_callback=lambda exc, name=fp.name: logging.error("Failed %s: %s", name, exc),
            )
        pool.close()
        pool.join()

    logging.info("All done.")


if __name__ == "__main__":
    main()
