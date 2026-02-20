# CLI Contracts & Function Signatures

As this feature is a batch CLI tool and not an API service, the "contracts" consist of the internal function signatures and the structure of the output summary log.

## 1. Output Summary Log (`ingestion_summary.csv`)
**Format**: CSV
**Headers**:
- `exam_name`: The original `fixed_name` identifier.
- `status`: `"success"`, `"failed"`, or `"skipped"`.
- `reason`: Explanation string (e.g., "Corrupted slice 4", "Missing PixelSpacing tag", "Already exists", or "").
- `output_path`: Absolute path to the `.nii.gz` file (empty if failed).

## 2. Core Conversion Function
**Signature**:
```python
def process_and_convert_exams(eligible_exams: list[dict], dicom_base_path: str, nifti_output_dir: str) -> list[dict]:
    """
    Orchestrates the conversion of DICOM series to NIfTI using MONAI.
    
    Args:
        eligible_exams: List of dictionaries containing exam metadata.
        dicom_base_path: Root path to the raw DICOM directories.
        nifti_output_dir: Root path where NIfTI outputs should be saved.
        
    Returns:
        A list of ConversionResult dictionaries (detailed in data-model.md) 
        used to generate the final CSV summary log.
    """
    pass
```

## 3. MONAI Pipeline Abstraction
**Signature**:
```python
def convert_series_to_nifti(dicom_dir: str, output_path: str) -> bool:
    """
    Uses MONAI LoadImage and SaveImage transforms to handle the actual 
    reading, affine parsing, Float32 normalization, and writing.
    
    Args:
        dicom_dir: Absolute path to the specific DICOM series directory.
        output_path: Absolute path to write the .nii.gz file.
        
    Returns:
        True if successful, raises an Exception (caught by orchestrator) if failed.
    """
    pass
```