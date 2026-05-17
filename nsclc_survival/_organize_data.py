#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DICOM data organization utility.

This script parses DICOM files within the raw data directory, retrieves 
metadata (PatientID and Modality) using pydicom, and moves the files into 
a new structured directory format organized by PatientID and Modality:
../data/organized_data/{PatientID}/{Modality}/

It also removes the original raw data folders after successful reorganization to save space.

"""

import shutil
import pydicom
from settings import RAW_DATA_PATH, ORGANIZED_DATA_PATH

ORGANIZED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Check whether the raw data folder exists or has already been removed (e.g. if the script is run twice)
if not RAW_DATA_PATH.exists():
    print(f"raw folder not found: probably all files have already been moved and the raw folder removed.")
    exit()

all_folders = [f for f in RAW_DATA_PATH.iterdir() if f.is_dir()]

print(f"Analysing {len(all_folders)} folders in progress...")

if len(all_folders) == 0:
    print("All folders have already been moved!")
else:
    # Reorganization cycle
    for folder_path in all_folders:
    
        # Find .dcm files in the folder
        files = list(folder_path.glob("*.dcm"))    
        if files:
            try:
                # Read the first file to get metadata
                sample_dcm = pydicom.dcmread(files[0])

                patient_id = sample_dcm.PatientID
                modality = sample_dcm.Modality
                # series_uid = sample_dcm.SeriesInstanceUID

                # Create new folder name for the target directory
                target_dir = ORGANIZED_DATA_PATH / patient_id / modality
                target_dir.mkdir(parents=True, exist_ok=True)

                # Move files to the new location
                for f in files:
                    shutil.move(str(f), str(target_dir / f.name))

                print(f"Moved: {patient_id} - {modality}")

            except Exception as e:
                print(f"Error processing {folder_path}: {e}")
                continue # if there's an error, don't delete the folder

        # Remove the original folder
        shutil.rmtree(folder_path)

    # Now remove also the raw folder
    try:
        shutil.rmtree(RAW_DATA_PATH)
        print(f"\nSuccessfully removed the entire '{RAW_DATA_PATH}' folder.")
    except Exception as e:
        print(f"Could not remove folder {RAW_DATA_PATH}: {e}")
        

    print("\nDone! Now all data are organized in '../data/organized_data' folder divided per patient")