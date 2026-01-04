import os
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib

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
                metadata_list.append(pydicom.dcmread(dcm_path, stop_before_pixels=True))
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
    Converts a sorted list of DICOM datasets to a NIfTI file.

    This function reads the full pixel data from the sorted DICOMs, stacks them
    into a 3D volume, and calculates the affine transformation matrix to ensure
    the NIfTI file has the correct orientation, spacing, and position.

    Args:
        sorted_dcms (list): A list of pydicom.Dataset objects, pre-sorted
                            in anatomical order (e.g., by filter_and_sort_dcms).
        output_nifti_path (str): The full path where the .nii.gz file will be saved.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    if not sorted_dcms:
        print("  - Error: Cannot convert to NIfTI, the provided DICOM list is empty.")
        return False

    # 1. Load pixel data and stack into a 3D array
    pixel_arrays = []
    for ds in sorted_dcms:
        try:
            # Re-read the file to get pixel data since we used stop_before_pixels earlier
            full_ds = pydicom.dcmread(ds.filename)
            pixel_arrays.append(full_ds.pixel_array)
        except Exception as e:
            print(f"  - Error: Could not read pixel data from {ds.filename}: {e}")
            return False

    # Stack along the 3rd axis (slice direction)
    try:
        volume_3d = np.stack(pixel_arrays, axis=-1)
    except ValueError:
        print("  - Error: Could not stack DICOMs. Slices may have different dimensions.")
        return False

    # 2. Calculate the Affine Transformation Matrix
    first_slice = sorted_dcms[0]
    last_slice = sorted_dcms[-1]

    # Get required DICOM tags with defaults
    iop = getattr(first_slice, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
    ipp = getattr(first_slice, 'ImagePositionPatient', [0, 0, 0])
    pixel_spacing = getattr(first_slice, 'PixelSpacing', [1.0, 1.0])

    # Calculate slice spacing from the sorted slice positions for robustness
    if len(sorted_dcms) > 1:
        slice_spacing = abs(last_slice.dist - first_slice.dist) / (len(sorted_dcms) - 1)
    else:
        slice_spacing = getattr(first_slice, 'SliceThickness', 1.0)

    # Define the affine matrix
    affine = np.identity(4)
    row_vec = np.array(iop[:3])
    col_vec = np.array(iop[3:])
    slice_vec = np.cross(row_vec, col_vec)

    affine[:3, 0] = row_vec * pixel_spacing[1]  # Column spacing
    affine[:3, 1] = col_vec * pixel_spacing[0]  # Row spacing
    affine[:3, 2] = slice_vec * slice_spacing
    affine[:3, 3] = ipp

    # 3. Create and save the NIfTI image
    nifti_img = nib.Nifti1Image(volume_3d, affine)
    nib.save(nifti_img, output_nifti_path)
    print(f"  - Successfully converted to {output_nifti_path}")
    return True

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