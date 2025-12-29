import os
import csv
import argparse
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
    parser = argparse.ArgumentParser(description="Extract DICOM metadata from all files in a folder to a CSV.")
    parser.add_argument("input_dir", help="Path to the directory containing DICOM files.")
    parser.add_argument("-o", "--output", dest="output_csv", default="dicom_metadata.csv",
                        help="Path for the output CSV file (default: dicom_metadata.csv).")
    
    args = parser.parse_args()
    
    extract_dicom_metadata_to_csv(args.input_dir, args.output_csv)