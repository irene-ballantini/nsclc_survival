#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DICOM data organization utility.

This module parses DICOM files within the raw data directory, retrieves 
metadata (PatientID and Modality) using pydicom, and moves the files into 
a new structured directory format organized by PatientID and Modality:

Example of the organized structure:
    organized_data/
        ├── PatientID_1/
        │   ├── CT/
        │   └── RTSTRUCT/
        ├── PatientID_2/
        │   ├── CT/
        │   └── RTSTRUCT/
        └── ...

It also removes the original raw data folders after successful reorganization to save space.

"""

import shutil
import pydicom
from pathlib import Path

def organize_dicom_data(raw_path, organized_path):
    """
    Reads DICOM files from raw_path, extracts PatientID and Modality,
    reorganizes them into organized_path/PatientID/Modality, and cleans up raw_path.
    
    Args:
        raw_path (Path or str): Folder path containing the downloaded data.
        organized_path (Path or str): Folder path containing data organized by patient ID.

    Raises:
        Exception: Logs errors if specific patient folders fail to process,
            preventing the final deletion of the raw directory to protect data.
    """
    raw_path = Path(raw_path)
    organized_path = Path(organized_path)

    organized_path.mkdir(parents=True, exist_ok=True)

    # Check whether the raw data folder exists or has already been removed (e.g. if the script is run twice)
    if not raw_path.exists():
        print(f"Warning: raw folder not found: probably all files have already been moved and the raw folder removed.")
        return

    all_folders = [f for f in raw_path.iterdir() if f.is_dir()]

    print(f"Analysing {len(all_folders)} folders in progress...")

    if len(all_folders) == 0:
        print("All folders have already been moved!")
    else:
        all_successful = True

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
                    target_dir = organized_path / patient_id / modality
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # Move files to the new location
                    for f in files:
                        shutil.move(str(f), str(target_dir / f.name))

                    print(f"Moved: {patient_id} - {modality}")

                    # Remove the original folder
                    shutil.rmtree(folder_path)

                except Exception as e:
                    print(f"Error processing {folder_path}: {e}")
                    all_successful = False
                    continue

            else:
                shutil.rmtree(folder_path)

        # Now remove also the raw folder if all folders have been moved successfully
        if all_successful:
            try:
                shutil.rmtree(raw_path)
                print(f"\nSuccessfully removed the entire '{raw_path}' folder.")
            except Exception as e:
                print(f"Could not remove folder {raw_path}: {e}")
        else:
            print(f"\nWarning: Some folders had errors. '{raw_path}' was NOT removed to protect data.")

    print(f"\nDone! Now all data are organized in '{organized_path}' folder divided per patient")
