"""
make_sample.py — build sample zips for pipeline testing.

Picks one class-0 and one class-1 exam per resolution group, plus QC edge-case
exams (class 2, empty folder, slice issues, etc.). Zips the raw DICOM folders
into /mnt/data/sample_zips/, grouped by their original zip source.

Requires a full pipeline run first (reads step1_convert.csv + raw DICOM folders).

Usage:
    python scripts/make_sample.py
"""
import argparse
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import config

SAMPLE_ROOT = config.SAMPLE_SOURCE_DIR


def _select_exams(report: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Pick representative + edge-case exams. Returns a DataFrame of selected rows."""
    exam_class = dict(zip(dataset["exam"], dataset["class"]))
    selected = []
    seen = set()

    def _add(row_or_id, group_id, cls, role):
        if isinstance(row_or_id, str):
            exam_id = row_or_id
            matches = report[report["exam_id"] == exam_id]
            if matches.empty:
                return
            row = matches.iloc[0]
        else:
            row = row_or_id
            exam_id = row["exam_id"]

        if exam_id in seen:
            return
        seen.add(exam_id)
        entry = row.to_dict()
        entry.update({"_group_id": group_id, "_role": role})
        selected.append(entry)
        print(f"  {role:25s}  {exam_id}  (group={group_id}, class={cls})")

    # 1. One class-0 and one class-1 per resolution group
    print("Resolution group representatives:")
    converted = report[report["conversion_status"] == "converted"].copy()
    for gid in range(1, 9):
        gdf = converted[converted["res_group"] == gid]
        for cls in [0.0, 1.0]:
            for _, row in gdf.iterrows():
                eid = row["exam_id"]
                if exam_class.get(eid) == cls:
                    _add(row, gid, int(cls), "group_repr")
                    break

    # 2. QC edge cases — one representative exam per case, picked from
    # either `dataset` (exam/class roster) or `report` (full QC columns).
    # "dataset" cases pass the exam ID (not the row) so _add looks the exam
    # up in `report` for its full column set.
    print("\nQC edge cases:")

    edge_cases = [
        # (source, filter_fn, cls_for_print, role)
        ("dataset", lambda df: df[df["class"] == 2.0], 2, "class_2"),
        ("report", lambda df: df[df["folder_status"] == "empty"], None, "empty_folder"),
        ("dataset", lambda df: df[df["class"].isna()], None, "no_class"),
        ("report", lambda df: df[df["folder_status"] == "review_duplicate"], None, "review_duplicate"),
        ("report", lambda df: df[(df["mixed_series"] == True) & (df["conversion_status"] == "converted")], None, "mixed_series"),
        ("report", lambda df: df[(df["scout_slices"] == True) & (df["conversion_status"] == "converted")], None, "scout_slices"),
        ("report", lambda df: df[df["qc_block_reason"] == "incomplete_metadata"], None, "incomplete_metadata"),
        ("report", lambda df: df[
            df["qc_block_reason"].str.contains(",", na=False) &
            ~df["qc_block_reason"].str.contains("excluded_class", na=False)
        ], None, "multi_reason_block"),
    ]
    for source, filter_fn, cls_hint, role in edge_cases:
        subset = filter_fn(dataset if source == "dataset" else report)
        if subset.empty:
            continue
        if source == "dataset":
            _add(subset.iloc[0]["exam"], None, cls_hint, role)
        else:
            _add(subset.iloc[0], None, cls_hint, role)

    # Distinct-value cases: one representative per unique issue value, not
    # just the first row — a single filter_fn/.iloc[0] wouldn't cover this.
    for _, row in report[report["slice_issue"].notna()].drop_duplicates("slice_issue").iterrows():
        _add(row, None, None, f"slice_{row['slice_issue']}")

    for _, row in report[report["extent_issue"].notna()].drop_duplicates("extent_issue").iterrows():
        _add(row, None, None, f"extent_{row['extent_issue']}")

    return pd.DataFrame(selected)


def _build_sample_dataset(selected: pd.DataFrame, staging_root: Path, out_root: Path) -> None:
    """Zip selected exam folders into out_root, grouped by zip_source."""
    if out_root.exists():
        # Preserve existing zips list for comparison
        old_zips = set(f.name for f in out_root.glob("*.zip"))
    else:
        old_zips = set()

    # Group exams by zip_source
    by_zip: dict[str, list[dict]] = defaultdict(list)
    for _, row in selected.iterrows():
        zs = row.get("zip_source")
        if pd.isna(zs):
            continue
        by_zip[str(zs)].append(row.to_dict())

    # Remove old output (except the zips we're about to recreate)
    for old in out_root.glob("*.zip"):
        old.unlink()

    out_root.mkdir(parents=True, exist_ok=True)

    # Re-batch into ~equal-sized zips named sample_batch_N.zip
    all_exams = []
    for zs, exams in sorted(by_zip.items()):
        for exam in exams:
            exam["_zip_source"] = zs
            all_exams.append(exam)

    batch_size = max(1, len(all_exams) // 3)  # aim for ~3 batches
    batches = []
    for i in range(0, len(all_exams), batch_size):
        batches.append(all_exams[i:i + batch_size])

    for batch_idx, batch in enumerate(batches, 1):
        zip_name = f"sample_batch_{batch_idx}.zip"
        zip_path = out_root / zip_name
        print(f"\n  Creating {zip_name} ({len(batch)} exams) ...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for exam in batch:
                original_name = exam.get("original_name")
                zs = exam["_zip_source"]
                exam_folder = staging_root / zs / original_name

                if not exam_folder.exists():
                    print(f"    WARNING: folder not found: {exam_folder}")
                    continue

                # Walk the exam folder and add all files
                for file_path in sorted(exam_folder.rglob("*")):
                    if file_path.is_file():
                        arcname = str(file_path.relative_to(staging_root / zs))
                        zf.write(file_path, arcname)

                print(f"    {original_name}")

    # Write manifest
    manifest = selected[["exam_id", "_group_id", "_role"]].copy()
    manifest.columns = ["exam_id", "group_id", "role"]
    manifest = manifest.sort_values("exam_id")
    manifest.to_csv(out_root / "manifest.csv", index=False)

    n_repr = (selected["_role"] == "group_repr").sum()
    n_edge = len(selected) - n_repr
    print(f"\nTotal: {len(selected)} exams  "
          f"(group representatives: {n_repr}, edge cases: {n_edge})")
    print(f"Output -> {out_root}  ({len(batches)} zip files)")


def main() -> None:
    if not config.STEP1_REPORT.exists():
        print(f"ERROR: {config.STEP1_REPORT} not found. Run the full pipeline first.")
        sys.exit(1)

    report = pd.read_csv(config.STEP1_REPORT)
    dataset = pd.read_csv(config.DATASET_CSV)

    selected = _select_exams(report, dataset)
    if selected.empty:
        print("No exams selected.")
        sys.exit(1)

    _build_sample_dataset(selected, config.STAGING_ROOT, SAMPLE_ROOT)


if __name__ == "__main__":
    main()
