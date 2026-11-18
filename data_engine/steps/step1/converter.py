"""Stage 6 — DICOM to NIfTI conversion.

Converts eligible DICOM series to NIfTI using MONAI LoadImage + ITKReader.
ITK handles series reconstruction, affine matrix construction, and DICOM edge
cases automatically.

Eligibility is determined entirely by qc_block_reason (set by stage 5).
Exams with a non-null qc_block_reason are skipped; all others are converted.
See qc.py:_compute_block_reason for the full list of blocking reasons.
Mixed-series exams use only the files belonging to selected_series_uid.
Non-ORIGINAL (scout/derived) files are excluded in all cases.

The DICOM loading and centroid-crop primitives are generic and live in
medprep.nifti_convert; this module supplies this project's target FOV
(config.TARGET_FOV_MM) and the per-exam orchestration (parallel workers,
skip/failure bookkeeping, output paths keyed by class/exam_id).
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import config
from medprep.nifti_convert import (
    centroid_crop_xy,
    load_dicom_series_as_array,
    resolve_dicom_file_list,
)

# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _convert_one(
    exam_id: str,
    dcm_path: str,
    selected_series_uid: str | None,
    output_path: str,
) -> dict[str, Any]:
    """Convert one DICOM series to NIfTI. Runs in a subprocess.

    Returns {"exam_id": ..., "conversion_status": ...}, plus "error": str(e)
    when conversion_status == "failed".
    OSError is re-raised so the caller can distinguish IO failure from
    DICOM parsing failure. All other exceptions are caught and reported as
    conversion_status = "failed".
    """
    import gc
    import nibabel as nib

    if Path(output_path).exists():
        return {"exam_id": exam_id, "conversion_status": "skipped_already_exists"}

    _tmpdir = None
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Always filter by series UID (for mixed-series) and ImageType (for
        # scouts). ITKReader requires a directory for 3D series reconstruction,
        # so when filtering reduces the file set we use a temp dir of symlinks.
        import tempfile, shutil
        file_list, _unreadable_count = resolve_dicom_file_list(Path(dcm_path), selected_series_uid)
        if not file_list:
            return {"exam_id": exam_id, "conversion_status": "failed"}

        all_files = sorted(f for f in Path(dcm_path).iterdir() if f.is_file())
        if len(file_list) == len(all_files) and selected_series_uid is None:
            # No filtering needed — pass directory directly.
            series_path = dcm_path
            _tmpdir = None
        else:
            # Scouts or mixed-series filtered out — symlink filtered files into
            # a temp directory so ITKReader gets the directory it expects.
            _tmpdir = tempfile.mkdtemp()
            for f in file_list:
                os.symlink(str(Path(f).resolve()), os.path.join(_tmpdir, Path(f).name))
            series_path = _tmpdir

        numpy_data, affine = load_dicom_series_as_array(series_path)

        numpy_data, affine, crop_x0, crop_y0 = centroid_crop_xy(
            numpy_data, affine, config.TARGET_FOV_MM
        )

        nib.save(nib.Nifti1Image(numpy_data, affine), output_path)

        del numpy_data
        gc.collect()

        return {
            "exam_id": exam_id,
            "conversion_status": "converted",
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
        }

    except OSError:
        raise
    except Exception as e:
        return {"exam_id": exam_id, "conversion_status": "failed", "error": str(e)}
    finally:
        if _tmpdir is not None:
            shutil.rmtree(_tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_conversion(
    df: pd.DataFrame,
    max_workers: int = 4,
    nifti_root: Path | None = None,
) -> pd.DataFrame:
    """Run stage 6: convert eligible exams to NIfTI.

    Adds conversion_status, crop_x0, crop_y0, and conversion_error columns to
    df and returns it. conversion_error is populated (str(e)) for failed
    exams and None otherwise. Skipped exams get an appropriate skipped_*
    status; never raises for individual exam failures (only for mount-point /
    IO problems).

    nifti_root: output directory for NIfTI files; defaults to config.NIFTI_ROOT.
                In test mode this is redirected to sample_nifti/ automatically.
    """
    if nifti_root is None:
        nifti_root = config.NIFTI_ROOT

    # Verify the output root is reachable before starting any conversion.
    try:
        nifti_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Cannot create NIfTI output directory {nifti_root}: {e}. "
            "Check that the target filesystem is mounted."
        ) from e

    results: dict[str, dict] = {}

    # Split into skipped (blocked) and eligible (to convert).
    tasks = []
    for _, row in df.iterrows():
        if pd.notna(row.get("qc_block_reason")):
            block_reason = str(row.get("qc_block_reason") or "")
            conv_status = (
                "skipped_excluded_class" if block_reason == "excluded_class"
                else "skipped"
            )
            results[row["exam_id"]] = {"conversion_status": conv_status}
            continue

        exam_id = row["exam_id"]
        dcm_path = row["dcm_path"]
        selected_series_uid = row.get("selected_series_uid")
        if pd.isna(selected_series_uid):
            selected_series_uid = None

        exam_class = int(row["class"])
        output_path = str(nifti_root / str(exam_class) / f"{exam_id}.nii.gz")
        tasks.append((exam_id, dcm_path, selected_series_uid, output_path))

    # Convert eligible exams in parallel.
    if tasks:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_convert_one, *t): t[0]
                for t in tasks
            }
            for future in as_completed(futures):
                exam_id = futures[future]
                try:
                    result = future.result()
                    results[result["exam_id"]] = result
                except OSError as e:
                    results[exam_id] = {"conversion_status": "failed", "error": str(e)}
                    print(f"  IO error converting {exam_id}: {e}")
                except Exception as e:
                    results[exam_id] = {"conversion_status": "failed", "error": str(e)}
                    print(f"  Unexpected error converting {exam_id}: {e}")

    df = df.copy()
    df["conversion_status"] = df["exam_id"].map(
        {e: r["conversion_status"] for e, r in results.items()}
    )
    df["crop_x0"] = df["exam_id"].map({e: r.get("crop_x0") for e, r in results.items()})
    df["crop_y0"] = df["exam_id"].map({e: r.get("crop_y0") for e, r in results.items()})
    df["conversion_error"] = df["exam_id"].map({e: r.get("error") for e, r in results.items()})
    return df
