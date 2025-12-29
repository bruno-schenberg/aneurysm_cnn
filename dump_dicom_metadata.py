import os
import csv
import pydicom
from pydicom.errors import InvalidDicomError

def extract_dicom_metadata_to_csv(input_dir, output_csv_path):
    """
    Reads all DICOM files in a directory, extracts their metadata,
    and writes it to a single CSV file.

    Args:
        input_dir (str): The path to the directory containing DICOM files.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    print(f"Scanning for DICOM files in '{input_dir}'...")
    dicom_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.dcm')]

    if not dicom_files:
        print("No DICOM (.dcm) files found.")
        return

    all_metadata = []
    all_fieldnames = set(['filename']) # Start with filename as a definite column

    print(f"Found {len(dicom_files)} files. Reading metadata...")

    # First pass: Read all files to gather data and all possible field names
    for f_path in dicom_files:
        try:
            ds = pydicom.dcmread(f_path, stop_before_pixels=True)
            
            # A dictionary to hold metadata for the current file
            file_metadata = {'filename': os.path.basename(f_path)}

            # Iterate through all data elements in the DICOM file
            for elem in ds:
                # Use the keyword as the column header. If no keyword, use the tag.
                key = elem.keyword if elem.keyword else str(elem.tag)
                
                # Handle different value types for clean CSV output
                if elem.value is None:
                    value = ''
                elif isinstance(elem.value, pydicom.multival.MultiValue):
                    # For multi-valued tags, join with a separator
                    value = ' | '.join(map(str, elem.value))
                else:
                    # For all other types, convert to string
                    value = str(elem.value)
                
                file_metadata[key] = value
                all_fieldnames.add(key)
            
            all_metadata.append(file_metadata)

        except InvalidDicomError:
            print(f"  - Warning: Skipping invalid DICOM file: {f_path}")
        except Exception as e:
            print(f"  - Error reading {f_path}: {e}")

    # Write to CSV
    if not all_metadata:
        print("Could not extract any metadata.")
        return

    # Sort fieldnames for consistent column order, with 'filename' first.
    sorted_fieldnames = sorted(list(all_fieldnames - {'filename'}))
    final_fieldnames = ['filename'] + sorted_fieldnames

    print(f"Writing metadata to '{output_csv_path}'...")
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames, restval='N/A')
            writer.writeheader()
            writer.writerows(all_metadata)
        print("Successfully created metadata CSV.")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # TODO: Set the path to the exam folder you want to process.
    INPUT_DIRECTORY = "/path/to/your/exam/folder"
    
    # You can also change the output file name if you wish.
    # A good practice is to name it after the folder.
    folder_name = os.path.basename(INPUT_DIRECTORY)
    OUTPUT_CSV_PATH = f"{folder_name}_metadata.csv"
    # --- End Configuration ---
    
    extract_dicom_metadata_to_csv(INPUT_DIRECTORY, OUTPUT_CSV_PATH)