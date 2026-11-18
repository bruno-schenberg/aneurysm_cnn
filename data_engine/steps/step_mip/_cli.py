"""Shared CLI plumbing for steps/step_mip/generate_*.py scripts.

Every generate_*.py script walks the same source_dir/<class>/ layout,
creates per-variant output directories, skips files whose outputs already
exist, and prints the same progress/summary format. This module factors
that out. Scripts that compute all their variants together for a file (and
skip the file only when every variant's output already exists) can use
run_variant_driver() directly; scripts with finer-grained (per-variant)
skip/save semantics use the lower-level helpers below instead, to preserve
their original resume behavior exactly.
"""
from pathlib import Path
from typing import Callable

import numpy as np

CLASSES = ("0", "1")


def collect_class_files(
    source_dir: Path,
    extensions: tuple[str, ...],
    classes: tuple[str, ...] = CLASSES,
    verbose: bool = True,
) -> list[tuple[Path, str]]:
    """Return (path, class) pairs for files under source_dir/<class>/ matching extensions.

    Always prints "WARNING: ... not found" for a missing class dir. When
    verbose (default), also prints "Class N: M volumes" per class, matching
    what the individual scripts used to print inline.
    """
    files = []
    for cls in classes:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} not found, skipping")
            continue
        cls_files = sorted(f for f in cls_dir.iterdir() if f.name.endswith(extensions))
        if verbose:
            print(f"Class {cls}: {len(cls_files)} volumes")
        files.extend((f, cls) for f in cls_files)
    return files


def apply_shard(
    items: list, shard_spec: str | None, show_total: bool = True,
) -> list:
    """Filter items to shard K of N given a 'K/N' spec string (no-op if None)."""
    if not shard_spec:
        return items
    k, n = map(int, shard_spec.split("/"))
    sharded = [it for i, it in enumerate(items) if i % n == k]
    if show_total:
        print(f"Shard {k}/{n}: processing {len(sharded)}/{len(items)} files")
    else:
        print(f"Shard {k}/{n}: processing {len(sharded)} files")
    return sharded


def file_stem(path: Path) -> str:
    """Strip a known NIfTI/.npy extension off a filename."""
    return path.name.replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")


def ensure_variant_dirs(
    output_root: Path, variant_names, classes: tuple[str, ...] = CLASSES,
) -> None:
    for variant in variant_names:
        for cls in classes:
            (output_root / variant / cls).mkdir(parents=True, exist_ok=True)


def variant_outputs_exist(
    output_root: Path, variant_names, cls: str, stem: str, ext: str = ".npy",
) -> bool:
    """True if every variant's output file for this (cls, stem) already exists."""
    return all((output_root / v / cls / f"{stem}{ext}").exists() for v in variant_names)


def save_variants(
    output_root: Path, cls: str, stem: str, arrays: dict[str, np.ndarray], ext: str = ".npy",
) -> None:
    for variant, arr in arrays.items():
        np.save(str(output_root / variant / cls / f"{stem}{ext}"), arr)


def count_variant_files(output_root: Path, variant_names) -> dict[str, int]:
    return {v: sum(1 for _ in (output_root / v).rglob("*.npy")) for v in variant_names}


def print_variant_counts(output_root: Path, variant_names) -> None:
    for variant, count in count_variant_files(output_root, variant_names).items():
        print(f"  {variant}: {count} files")


def run_variant_driver(
    source_dir: Path,
    output_root: Path,
    variant_names,
    process_fn: Callable[[Path], dict[str, np.ndarray]],
    file_ext: tuple[str, ...] = (".nii", ".nii.gz"),
    classes: tuple[str, ...] = CLASSES,
    progress_every: int = 50,
    track_errors: bool = False,
    skipped_label: str = "Skipped",
) -> None:
    """Common driver for scripts that compute all of a file's variants together.

    process_fn(path) -> {variant_name: array}. A file is skipped only when
    every variant's output already exists — recomputing is all-or-nothing
    per file, matching generate_mips.py's/generate_lausanne_mips.py's
    original behavior (cheap to recompute all variants together since the
    volume is only loaded once).
    """
    ensure_variant_dirs(output_root, variant_names, classes)

    total_generated = 0
    total_skipped = 0
    total_errors = 0

    for cls in classes:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} not found, skipping")
            continue

        files = sorted(f for f in cls_dir.iterdir() if f.name.endswith(file_ext))
        print(f"Class {cls}: {len(files)} volumes")

        for i, fpath in enumerate(files):
            stem = file_stem(fpath)

            if variant_outputs_exist(output_root, variant_names, cls, stem):
                total_skipped += 1
                continue

            if track_errors:
                try:
                    arrays = process_fn(fpath)
                except Exception as e:
                    print(f"  ERROR processing {fpath.name}: {e}")
                    total_errors += 1
                    continue
            else:
                arrays = process_fn(fpath)

            save_variants(output_root, cls, stem, arrays)
            total_generated += 1

            if (i + 1) % progress_every == 0 or (i + 1) == len(files):
                print(f"  [{i + 1}/{len(files)}] {stem}")

    summary = f"\nDone. Generated: {total_generated}, {skipped_label}: {total_skipped}"
    if track_errors:
        summary += f", Errors: {total_errors}"
    print(summary)
    print_variant_counts(output_root, variant_names)
