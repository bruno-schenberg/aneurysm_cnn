"""Are the final-campaign datasets ready? Exit 0 = READY.

Checks every dataset that configs/final/*.json reference, plus every variant
folder that exists in the MIP families:
  - exam IDs + labels match the source NIfTIs (ours or Lausanne), minus
    dataset/exclusions.json; no missing, extra or mislabeled exams
  - each file loads, is finite, not constant (each MIP view too), no float16
    overflow; one shape per folder; no duplicate arrays (except excluded copies)
  - mixed folders: every symlink resolves
  - the frozen test exams are present, and the final split builds (patient
    groups, no patient in both test and dev)
--quick skips loading files (IDs, labels and the split only, seconds).

Usage (from training_engine/):
    python scripts/check_datasets.py [--quick] [--workers 6]
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402
from analysis.splits import label_only_folds  # noqa: E402
from src.split import exam_key, load_exclusions  # noqa: E402

OURS, LAUS = Path("/mnt/data/nifti"), Path("/mnt/data/lausanne_nifti")
EXTRA_FAMILIES = ("mip_grid_n4_lausanne",)
EXTS = (".npy", ".nii.gz")


def labels_of(root: Path, prefix: str = "") -> dict[str, int]:
    return {prefix + f.name[:-7]: int(c) for c in ("0", "1") for f in (root / c).glob("*.nii.gz")}


def listing(d: Path) -> dict[str, tuple[int, Path]]:
    return {exam_key(str(f)): (int(c), f) for c in ("0", "1") for f in (d / c).iterdir()
            if f.name.endswith(EXTS)}


def check_file(path: str):
    try:
        a = np.load(path) if path.endswith(".npy") else np.asarray(nib.load(path).dataobj)
    except Exception as e:
        return path, None, [f"load failed: {e}"], None
    a32 = a.astype(np.float32)
    bad = []
    if not np.isfinite(a32).all():
        bad.append("non-finite values")
    if a32.max() == a32.min():
        bad.append("constant")
    elif a.ndim == 3 and a.shape[0] == 3:  # MIP: every view must have content
        bad += [f"view {v} constant" for v in range(3) if a32[v].max() == a32[v].min()]
    if a.dtype == np.float16 and (np.abs(a32) >= 65504).any():
        bad.append("float16 overflow")
    return path, (a.shape, str(a.dtype)), bad, hashlib.md5(a.tobytes()).hexdigest()


def targets() -> tuple[list[Path], list[str]]:
    """(dataset dirs to check, referenced-but-missing data_path_keys)."""
    dirs, missing = set(), []
    for f in sorted((ROOT / "configs" / "final").glob("*.json")):
        key = json.loads(f.read_text())[0]["data_path_key"]
        path = Path(cfg.DATASET_PATHS[key])
        (dirs.add(path) if path.exists() else missing.append(f"{key} (by {f.stem})"))
    for fam in (*cfg.MIP_GRID_FAMILIES, *EXTRA_FAMILIES):
        fam_dir = Path(cfg.DATASET_PATHS["iso_256"]).parent / fam
        if fam_dir.exists():
            dirs |= {p for p in fam_dir.iterdir() if p.is_dir()}
    return sorted(dirs), sorted(set(missing))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true", help="IDs/labels/split only, no file loading")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    excluded, patients = load_exclusions("dataset/exclusions.json")
    ours, laus = labels_of(OURS), labels_of(LAUS, "laus_")
    problems: list[str] = []
    unknown = excluded - set(ours) - set(laus)
    if unknown:
        problems.append(f"exclusions.json lists unknown exams: {sorted(unknown)}")
    unknown = set(patients) - set(ours) - set(laus)
    if unknown:
        problems.append(f"exclusions.json patient_groups list unknown exams: {sorted(unknown)}")
    try:
        test_ids, folds = label_only_folds(ours)
        print(f"split: {len(test_ids)} test, {sum(map(len, folds))} dev in {len(folds)} folds "
              f"({len(excluded & set(ours))} of ours excluded)")
    except (AssertionError, ValueError) as e:
        test_ids = set()
        problems.append(f"final split does not build: {e}")

    dirs, missing = targets()
    problems += [f"referenced dataset not built: {m}" for m in missing]
    jobs = []
    for d in dirs:
        ref = {**ours, **laus} if "mixed" in d.parent.name else laus if "lausanne" in d.parent.name else ours
        found = listing(d)
        want = {k: v for k, v in ref.items() if k not in excluded}
        name = d.relative_to(d.parents[1]) if d.parent.name.startswith("mip_grid") else d.name
        lost = sorted(set(want) - set(found))
        extra = sorted(set(found) - set(ref))
        wrong = sorted(k for k in set(want) & set(found) if want[k] != found[k][0])
        if "laus" not in str(d) and not test_ids <= set(found):
            problems.append(f"{name}: test exams missing")
        for what, ids in (("missing", lost), ("unexpected", extra), ("wrong label", wrong)):
            if ids:
                problems.append(f"{name}: {len(ids)} {what}, e.g. {ids[:5]}")
        if "mixed" in d.parent.name:
            broken = [f.name for _, f in found.values() if not f.exists()]
            if broken:
                problems.append(f"{name}: {len(broken)} broken symlinks, e.g. {broken[:3]}")
        elif not args.quick:
            jobs += [(str(name), k in excluded, str(f)) for k, (_, f) in found.items()]
    print(f"{len(dirs)} dataset folders: IDs/labels checked"
          + ("" if args.quick else f"; loading {len(jobs)} files"))

    shapes, hashes = defaultdict(Counter), defaultdict(list)
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(check_file, [f for _, _, f in jobs], chunksize=32)
        for (name, is_excluded, _), (path, shape, bad, h) in zip(jobs, results):
            if is_excluded:
                continue  # excluded copies may duplicate their originals or be broken
            problems += [f"{name}: {Path(path).name}: {b}" for b in bad]
            shapes[name][shape] += 1
            hashes[(name, h)].append(Path(path).name)
    for name, c in shapes.items():
        if len(c) > 1:
            problems.append(f"{name}: {len(c)} different shapes {dict(c)}")
    problems += [f"{n}: identical arrays {v}" for (n, _), v in hashes.items() if len(v) > 1]

    for line in problems:
        print("  PROBLEM", line)
    print("NOT READY" if problems else "READY", f"({len(problems)} problems)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
