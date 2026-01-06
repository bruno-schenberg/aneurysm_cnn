import os
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import dicom2nifti
import dicom2nifti.settings as settings

# Optional: Disable validation if you trust your 'OK' status from earlier
# settings.disable_validate_slice_increment() 

def filter_for_conversion(exam_data):
    """
    Filters a list of exams for those ready for NIfTI conversion.

    An exam is considered ready if its 'validation_status' is 'OK' and its
    'class' is either '0' or '1'. These are the cases that have been
    successfully validated and have a definitive classification.

    Args:
        exam_data (list): A list of dictionaries, where each dictionary
                          represents a processed exam.

    Returns:
        list: A list of dictionaries, where each dictionary represents an exam
              that meets the criteria. Returns an empty list if no exams
              meet the criteria.
    """
    print("\nFiltering exams for NIfTI conversion...")

    eligible_exams = [
        exam for exam in exam_data
        if exam.get('validation_status') == 'OK' and exam.get('class') in ('0', '1')
    ]

    print(f"  - Found {len(eligible_exams)} exams eligible for conversion.")
    return eligible_exams

def filter_and_sort_dcms(input_path):
    """
    Loads, filters, and sorts DICOM files from a directory.

    1. Loads all DICOM metadata from the given path.
    2. Filters out scout/localizer images, keeping only 'ORIGINAL' images.
    3. Sorts the remaining images using projection-based sorting to ensure
       correct anatomical order.

    Args:
        input_path (str): The full path to the directory containing .dcm files.

    Returns:
        list: A sorted list of pydicom dataset objects for the main series,
              ready for further processing like NIfTI conversion. Returns an
              empty list if no valid DICOMs are found.
    """
    # Load all DICOM metadata from the directory
    metadata_list = []
    for f in os.listdir(input_path):
        if f.lower().endswith('.dcm'):
            try:
                dcm_path = os.path.join(input_path, f)
                metadata_list.append(pydicom.dcmread(dcm_path))
            except InvalidDicomError:
                print(f"  - Warning: Skipping invalid DICOM file: {dcm_path}")

    # Filter for main series images (exclude scouts/localizers)
    main_series_metadata = [ds for ds in metadata_list if 'ORIGINAL' in getattr(ds, 'ImageType', [])]

    if not main_series_metadata:
        return []

    # Projection-Based Sorting
    sample_ds = main_series_metadata[0]
    iop = getattr(sample_ds, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
    normal_vec = np.cross(np.array(iop[:3]), np.array(iop[3:]))
    for ds in main_series_metadata:
        ds.dist = np.dot(np.array(getattr(ds, 'ImagePositionPatient', [0, 0, 0])), normal_vec)
    main_series_metadata.sort(key=lambda x: x.dist)

    return main_series_metadata

def convert_dcms_to_nifti(sorted_dcms, output_nifti_path):
    """
    Converts a sorted list of DICOM datasets to a NIfTI file using dicom2nifti.
    """
    if not sorted_dcms:
        print("  - Error: Cannot convert to NIfTI, the provided DICOM list is empty.")
        return False

    try:
        # dicom2nifti can take a list of pydicom datasets directly.
        # It handles orientation, scaling, and stacking for you.
        dicom2nifti.convert_dicom.dicom_array_to_nifti(
            sorted_dcms, 
            output_nifti_path, 
            reorient_nifti=True
        )
        print(f"  - Successfully converted to {output_nifti_path}")
        return True
        
    except Exception as e:
        print(f"  - Error during conversion for {output_nifti_path}: {e}")
        return False
    
def process_and_convert_exams(eligible_exams, dicom_base_path, nifti_output_dir):
    """
    Orchestrates the conversion of eligible DICOM series to NIfTI format.

    This function iterates through a list of eligible exams, and for each one,
    it calls `filter_and_sort_dcms` to prepare the DICOM files and then
    `convert_dcms_to_nifti` to perform the conversion.

    Args:
        eligible_exams (list): A list of exam dictionaries, typically from
                               `filter_for_conversion()`.
        dicom_base_path (str): The base directory where the raw DICOM data is stored
                               (e.g., '/mnt/data/cases-3/raw').
        nifti_output_dir (str): The directory where the output .nii.gz files
                                will be saved.
    """
    print(f"\nStarting NIfTI conversion for {len(eligible_exams)} exams...")
    if not os.path.exists(nifti_output_dir):
        print(f"  - Creating base output directory: {nifti_output_dir}")
        os.makedirs(nifti_output_dir)

    converted_count = 0
    for exam in eligible_exams:
        fixed_name = exam['fixed_name']
        data_path = exam['data_path']
        exam_class = exam.get('class')

        # The filter_for_conversion function ensures class is '0' or '1'
        print(f"\nProcessing exam: {fixed_name} (Class: {exam_class})")

        # Create the class-specific subdirectory (e.g., .../nifti/0/)
        class_output_dir = os.path.join(nifti_output_dir, str(exam_class))
        os.makedirs(class_output_dir, exist_ok=True)

        input_dicom_path = os.path.join(dicom_base_path, data_path)
        output_nifti_path = os.path.join(class_output_dir, f"{fixed_name}.nii.gz")

        # Step 1: Filter and sort the DICOM series from the input path
        sorted_dcms = filter_and_sort_dcms(input_dicom_path)

        # Step 2: Convert the sorted series to a NIfTI file
        if convert_dcms_to_nifti(sorted_dcms, output_nifti_path):
            converted_count += 1

    print(f"\nConversion complete. Successfully converted {converted_count} out of {len(eligible_exams)} exams.")