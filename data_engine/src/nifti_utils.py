"""
nifti_utils.py

Converts validated DICOM series into NIfTI volumes and saves them in the
class-labelled directory layout expected by the training engine.

## What is NIfTI and why convert to it?

DICOM (Digital Imaging and Communications in Medicine) is the format hospitals
use to store and transmit medical images. It is powerful but complex: a single
scan is stored as hundreds of individual files (one per slice), each carrying
its own metadata. Working directly with DICOM during training would be slow and
fragile.

NIfTI (Neuroimaging Informatics Technology Initiative) is the standard format
used in research pipelines. It stores the entire 3D volume in a single
compressed file (``.nii.gz``) alongside an affine matrix that encodes the
real-world position and orientation of the volume in millimetres. MONAI, the
medical imaging deep learning framework used for training, reads NIfTI natively
and efficiently.

## Output layout

The training engine's data loader expects NIfTI files organised by class label:

  nifti/
  ├── 0/          ← healthy cases (no aneurysm)
  │   ├── BP001.nii.gz
  │   ├── BP004.nii.gz
  │   └── ...
  └── 1/          ← aneurysm cases
      ├── BP002.nii.gz
      ├── BP003.nii.gz
      └── ...

This module creates that layout automatically from the ``class`` field attached
to each case record by ``class_utils``.

## Parallelism

Converting a large dataset (hundreds of cases) is CPU- and memory-intensive.
This module uses Python's ``ProcessPoolExecutor`` to convert multiple cases
concurrently, one case per subprocess. True subprocess parallelism is used
rather than threads because MONAI and ITK are C++ libraries that manage their
own internal thread pools — mixing them with Python threads can cause contention.
Each subprocess is entirely independent and communicates its result back to the
main process as a plain dictionary.

Public API (called by data_cleaner.py):
  - filter_for_conversion(exam_data)           : select cases eligible for conversion
  - process_and_convert_exams(...)             : run parallel conversion and return results
"""

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
    Selects only the cases that are ready and safe to convert to NIfTI.

    Two conditions must both be true for a case to be included:

    1. ``validation_status == 'OK'`` — the DICOM series passed all geometric
       integrity checks in ``dicom_utils.validate_dcms``. Cases with any other
       status (e.g. ``VARIABLE_SPACING``, ``MIXED_SERIES_ERROR``, ``BELOW_LIMIT``)
       are excluded because their volumes may be corrupted or unreliable.

    2. ``class in ('0', '1')`` — the case has a confirmed ground-truth label
       from ``classes.csv``. Cases without a label cannot be used for supervised
       training and must not be placed in the output directory, where their
       presence would silently introduce unlabelled data into the training set.

    Args:
        exam_data: Full list of case record dicts after validation and class
                   joining have both been completed.

    Returns:
        A filtered list containing only cases that meet both conditions above.
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
    Loads a DICOM series from a directory and saves it as a compressed NIfTI file.

    **Why MONAI + ITKReader?**
    MONAI's ``LoadImage`` with ``ITKReader`` is used rather than a raw pydicom
    approach because ITK (Insight Toolkit) has been the gold standard for medical
    image I/O for over two decades. It handles all the complex details of DICOM
    series reconstruction automatically: stacking slices in the correct order,
    constructing the affine matrix from DICOM spatial tags, and handling edge
    cases in multi-frame and enhanced DICOM formats. Writing this logic manually
    using pydicom would be error-prone and fragile.

    **What is the affine matrix?**
    The 3D volume itself is just a grid of numbers — it has no inherent sense of
    scale, position, or orientation. Two scans with the same pixel dimensions
    could be a paediatric head or an adult abdomen; you cannot tell from the
    voxel values alone.

    The affine is a small lookup table (separate from the volume) that answers
    the question: "given a voxel at position (x, y, z) in the grid, where is
    that point in the real world, in millimetres, relative to the patient?" It
    encodes three things: the physical size of each voxel (e.g. 0.5 mm × 0.5 mm
    × 1 mm), which direction each axis points relative to the patient's body, and
    where the origin of the grid sits in space.

    NIfTI stores this table alongside the pixel data so that any tool reading
    the file — whether it is a visualiser, a registration algorithm, or the
    training pipeline — knows exactly where each voxel sits anatomically. ITK
    reconstructs it automatically from the DICOM spatial tags
    (ImagePositionPatient, ImageOrientationPatient, PixelSpacing).

    **Float32 casting:**
    DICOM pixel values are typically stored as 16-bit integers. They are cast to
    32-bit floating point here so the NIfTI file is in the dtype that PyTorch and
    MONAI transforms expect during training. Storing as float32 also standardises
    the format across scanners that may use different integer precisions.

    **Memory management:**
    ITK and MONAI can retain large internal allocations after processing. The
    loaded image, numpy array, and NIfTI object are explicitly deleted and
    ``gc.collect()`` is called immediately after saving. This prevents memory
    from accumulating across conversions when many cases are processed
    sequentially within a single subprocess.

    **Error handling:**
    ``OSError`` (e.g. disk full, permission denied) is re-raised rather than
    caught so the caller can distinguish a disk I/O failure from a DICOM parsing
    failure. All other exceptions are caught, logged, and indicated via a
    ``False`` return value so a single bad case does not abort the whole batch.

    Args:
        dicom_dir: Absolute path to the directory containing the ``.dcm`` files
                   for this series.
        output_path: Absolute path where the ``.nii.gz`` file will be written.

    Returns:
        ``True`` if conversion and save succeeded, ``False`` otherwise.

    Raises:
        OSError: If the file cannot be written (disk full, permission denied, etc.).
    """
    try:
        # ITKReader handles the full DICOM series → 3D volume reconstruction,
        # including affine matrix construction from spatial DICOM tags.
        # image_only=True returns just the MetaTensor (no metadata dict wrapper).
        loader = LoadImage(reader=ITKReader(), image_only=True)
        image_data = loader(dicom_dir)

        # Cast to float32: standardises dtype across scanners and matches
        # what the training pipeline expects.
        numpy_data = image_data.numpy().astype(np.float32)
        # Preserve the affine matrix so spatial information survives conversion.
        affine = image_data.affine.numpy()

        nifti_img = nib.Nifti1Image(numpy_data, affine)

        # OSError (disk full, permission denied) is intentionally NOT caught here
        # so the caller can distinguish IO failures from conversion failures.
        nib.save(nifti_img, output_path)

        logger.info(f"  - Successfully converted {dicom_dir} to {output_path}")

        # Explicitly release large objects and force garbage collection.
        # ITK/MONAI can hold memory in C++ heap that Python's GC does not see.
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
    Converts a single exam from DICOM to NIfTI. Designed to run in a subprocess.

    This function is the unit of work dispatched to each parallel worker by
    ``process_and_convert_exams``. Because it runs in a separate process, it
    cannot write to the main process's logger directly — doing so would either
    produce garbled output (multiple processes writing to the same file handle)
    or be silently lost. Instead, log messages are collected in a list inside
    the result dict and flushed by the main process after each future completes.

    **Output path:**
    The NIfTI file is written to ``{nifti_output_dir}/{class}/{fixed_name}.nii.gz``.
    The class subdirectory (``0/`` or ``1/``) is created automatically if it does
    not exist. The filename uses the canonical ``fixed_name`` (e.g. ``BP001``),
    not the original raw folder name, so filenames are consistent regardless of
    how the source data was named.

    **Idempotency:**
    If the output file already exists, the case is skipped and marked
    ``'skipped'`` rather than re-converted. This makes the pipeline safe to
    re-run after an interruption — only cases that did not complete will be
    processed, leaving successfully converted files untouched.

    **Error separation:**
    ``OSError`` (disk/IO failures) and general exceptions are caught separately
    and assigned distinct reason strings. This makes it easier to diagnose
    whether a failure was caused by a full disk or a bad DICOM file.

    Args:
        exam: Case record dict with at minimum ``'fixed_name'``, ``'class'``,
              and ``'data_path'`` keys.
        dicom_base_path: Absolute path to the raw DICOM root directory.
        nifti_output_dir: Absolute path to the NIfTI output root directory.

    Returns:
        A result dict with keys:
          ``'exam_name'``     — canonical case name
          ``'status'``        — ``'success'``, ``'skipped'``, or ``'failed'``
          ``'reason'``        — empty string on success; error description on failure
          ``'output_path'``   — absolute path to the output file (empty on failure)
          ``'log_messages'``  — list of log strings to be flushed by the main process
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

    # Create the class subdirectory (0/ or 1/) if it does not already exist.
    class_output_dir = os.path.join(nifti_output_dir, str(exam_class))
    os.makedirs(class_output_dir, exist_ok=True)

    input_dicom_path = os.path.join(dicom_base_path, exam['data_path'])
    output_nifti_path = os.path.join(class_output_dir, f"{fixed_name}.nii.gz")

    # Skip if output already exists — makes the pipeline safely re-runnable.
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
    Converts all eligible DICOM series to NIfTI in parallel using subprocesses.

    Submits one ``_convert_exam_worker`` call per exam to a ``ProcessPoolExecutor``
    and collects results as each future completes. Using separate processes rather
    than threads avoids GIL contention and the internal thread-pool conflicts that
    MONAI and ITK can exhibit when run from Python threads.

    ``as_completed`` is used instead of iterating futures in submission order so
    that results are logged and collected as soon as each exam finishes, giving
    real-time progress feedback during long conversion runs.

    After all futures complete, log messages bundled inside each result are
    flushed to the main process logger and then removed from the result dict
    before it is appended to the output list.

    Args:
        eligible_exams: List of case record dicts as returned by
                        ``filter_for_conversion``.
        dicom_base_path: Absolute path to the raw DICOM root directory.
        nifti_output_dir: Absolute path to the NIfTI output root directory.
                          Created automatically if it does not exist.
        max_workers: Number of parallel worker processes. ``None`` defaults to
                     ``os.cpu_count()``. Reduce this if memory is constrained,
                     since each worker loads a full 3D volume into RAM.

    Returns:
        List of result dicts, one per exam, each with keys:
        ``'exam_name'``, ``'status'``, ``'reason'``, ``'output_path'``.
        The ``'log_messages'`` key is removed before returning.
    """
    logger.info(f"Starting NIfTI conversion for {len(eligible_exams)} exams...")
    os.makedirs(nifti_output_dir, exist_ok=True)

    results = []
    futures = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for exam in eligible_exams:
            future = executor.submit(_convert_exam_worker, exam, dicom_base_path, nifti_output_dir)
            # Map future → name so we can identify which exam failed if an
            # unhandled exception propagates out of the worker.
            futures[future] = exam['fixed_name']

        for future in as_completed(futures):
            result = future.result()
            # Flush worker log messages into the main process logger.
            for msg in result.pop("log_messages", []):
                logger.info(msg)
            results.append(result)

    success_count = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"Conversion complete. Successfully processed {success_count} out of {len(eligible_exams)} exams.")
    return results
