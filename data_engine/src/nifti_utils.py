import os
import logging
import numpy as np
import nibabel as nib
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from monai.transforms import LoadImage
from monai.data import ITKReader

logger = logging.getLogger("dicom_ingestion")


# ----------------------------------------------------
# 1. Pre-Conversion Filtering
# ----------------------------------------------------


def filter_for_conversion(exam_data: list[dict]) -> list[dict]:
    """
    Filters a list of exams for those ready for NIfTI conversion.
    """
    logger.info("Filtering exams for NIfTI conversion...")

    eligible_exams = [
        exam for exam in exam_data
        if exam.get('validation_status') == 'OK' and exam.get('class') in ('0', '1')
    ]

    logger.info(f"  - Found {len(eligible_exams)} exams eligible for conversion.")
    return eligible_exams


# ----------------------------------------------------
# 2. NIfTI Conversion
# ----------------------------------------------------


def convert_series_to_nifti(dicom_dir: str, output_path: str) -> bool:
    """
    Uses MONAI to load a DICOM series and save it as a NIfTI file.
    Ensures Float32 normalization and affine preservation.
    """
    try:
        # MONAI LoadImage with ITKReader is generally very robust for spatial metadata
        # It handles the affine matrix calculation from DICOM tags automatically.
        loader = LoadImage(reader=ITKReader(), image_only=True)
        # Load the directory
        image_data = loader(dicom_dir)

        # Ensure data type is Float32
        # MONAI images (MetaTensor) can be converted to numpy with specific type
        numpy_data = image_data.numpy().astype(np.float32)
        affine = image_data.affine.numpy()

        # Create NiBabel image
        nifti_img = nib.Nifti1Image(numpy_data, affine)

        # Save to disk — OSError (e.g. disk full) is intentionally NOT caught here
        # so callers can distinguish IO failures from conversion failures (FR-014)
        nib.save(nifti_img, output_path)

        logger.info(f"  - Successfully converted {dicom_dir} to {output_path}")

        # Explicit garbage collection to free memory before processing the next exam
        del image_data, numpy_data, nifti_img
        gc.collect()

        return True
    except OSError:
        raise
    except Exception as e:
        logger.error(f"  - Error during MONAI conversion for {dicom_dir}: {e}")
        return False


# ----------------------------------------------------
# 3. Conversion Orchestration
# ----------------------------------------------------


def _convert_exam_worker(exam: dict, dicom_base_path: str, nifti_output_dir: str) -> dict:
    """
    Worker function for parallel NIfTI conversion of a single exam.
    Designed to run in a subprocess — returns all results and log messages
    as data so the main process can log them.
    """
    fixed_name = exam['fixed_name']
    exam_class = exam.get('class')

    result = {
        "exam_name": fixed_name,
        "status": "failed",
        "reason": "",
        "output_path": "",
        "log_messages": [],
    }

    result["log_messages"].append(f"Processing exam: {fixed_name} (Class: {exam_class})")

    class_output_dir = os.path.join(nifti_output_dir, str(exam_class))
    os.makedirs(class_output_dir, exist_ok=True)

    input_dicom_path = os.path.join(dicom_base_path, exam['data_path'])
    output_nifti_path = os.path.join(class_output_dir, f"{fixed_name}.nii.gz")

    if os.path.exists(output_nifti_path):
        result["log_messages"].append(f"  - Skipping {fixed_name}, output already exists")
        result["status"] = "skipped"
        result["reason"] = "Already exists"
        result["output_path"] = output_nifti_path
        return result

    try:
        if convert_series_to_nifti(input_dicom_path, output_nifti_path):
            result["status"] = "success"
            result["output_path"] = output_nifti_path
        else:
            result["reason"] = "MONAI conversion failed"
    except OSError as e:
        result["reason"] = "FAILED_IO"
        result["log_messages"].append(f"  - IO failure for {fixed_name}: {e}")
    except Exception as e:
        result["reason"] = f"Unexpected error: {str(e)}"

    return result


def process_and_convert_exams(
    eligible_exams: list[dict],
    dicom_base_path: str,
    nifti_output_dir: str,
    max_workers: int | None = None,
) -> list[dict]:
    """
    Orchestrates parallel conversion of eligible DICOM series to NIfTI format.
    Uses a ProcessPoolExecutor so each exam is converted concurrently.

    Args:
        eligible_exams: Exams that passed validation and have a known class.
        dicom_base_path: Root directory containing the raw DICOM folders.
        nifti_output_dir: Root directory for NIfTI output (class subdirs created automatically).
        max_workers: Number of parallel worker processes. Defaults to os.cpu_count().

    Returns:
        A list of ConversionResult dicts (one per exam).
    """
    logger.info(f"Starting NIfTI conversion for {len(eligible_exams)} exams...")
    os.makedirs(nifti_output_dir, exist_ok=True)

    results = []
    futures = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for exam in eligible_exams:
            future = executor.submit(_convert_exam_worker, exam, dicom_base_path, nifti_output_dir)
            futures[future] = exam['fixed_name']

        for future in as_completed(futures):
            result = future.result()
            for msg in result.pop("log_messages", []):
                logger.info(msg)
            results.append(result)

    success_count = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"Conversion complete. Successfully processed {success_count} out of {len(eligible_exams)} exams.")
    return results
