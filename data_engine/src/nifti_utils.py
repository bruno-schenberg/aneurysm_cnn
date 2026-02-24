import os
import logging
import numpy as np
import nibabel as nib
import gc
from monai.transforms import LoadImage
from monai.data import ITKReader, PydicomReader

logger = logging.getLogger("dicom_ingestion")

def filter_for_conversion(exam_data):
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

def convert_series_to_nifti(dicom_dir, output_path):
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
        
        # Save to disk
        nib.save(nifti_img, output_path)
        
        logger.info(f"  - Successfully converted {dicom_dir} to {output_path}")
        
        # Explicit garbage collection to free memory before processing the next exam
        del image_data, numpy_data, nifti_img
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"  - Error during MONAI conversion for {dicom_dir}: {e}")
        return False

def process_and_convert_exams(eligible_exams, dicom_base_path, nifti_output_dir):
    """
    Orchestrates the conversion of eligible DICOM series to NIfTI format.
    Returns a list of ConversionResult dictionaries.
    """
    results = []
    logger.info(f"Starting NIfTI conversion for {len(eligible_exams)} exams...")
    if not os.path.exists(nifti_output_dir):
        logger.info(f"  - Creating base output directory: {nifti_output_dir}")
        os.makedirs(nifti_output_dir)

    for exam in eligible_exams:
        fixed_name = exam['fixed_name']
        data_path = exam['data_path']
        exam_class = exam.get('class')

        result = {
            "exam_name": fixed_name,
            "status": "failed",
            "reason": "",
            "output_path": ""
        }

        logger.info(f"Processing exam: {fixed_name} (Class: {exam_class})")

        # Create the class-specific subdirectory
        class_output_dir = os.path.join(nifti_output_dir, str(exam_class))
        os.makedirs(class_output_dir, exist_ok=True)

        input_dicom_path = os.path.join(dicom_base_path, data_path)
        output_nifti_path = os.path.join(class_output_dir, f"{fixed_name}.nii.gz")

        # Idempotency check: skip if output already exists
        if os.path.exists(output_nifti_path):
            logger.info(f"  - Skipping {fixed_name}, output already exists: {output_nifti_path}")
            result["status"] = "skipped"
            result["reason"] = "Already exists"
            result["output_path"] = output_nifti_path
            results.append(result)
            continue

        # Step 1: Convert using the new MONAI-based function
        try:
            if convert_series_to_nifti(input_dicom_path, output_nifti_path):
                result["status"] = "success"
                result["output_path"] = output_nifti_path
            else:
                result["reason"] = "MONAI conversion failed"
        except Exception as e:
            result["reason"] = f"Unexpected error: {str(e)}"
        
        results.append(result)

    success_count = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"Conversion complete. Successfully processed {success_count} out of {len(eligible_exams)} exams.")
    return results
