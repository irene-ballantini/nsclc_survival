#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Module for downloading NSCLC-Radiomics dataset from TCIA. 

"""

from tcia_utils import nbia
from nsclc_survival.settings import RAW_DATA_PATH, COLLECTION_NAME, N_PATIENTS, patientID, modality, CT, RTSTRUCT

import logging

logger = logging.getLogger(__name__)

def download_nsclc_radiomics_data(n_patients_to_download=N_PATIENTS):
    """
    This function connects to the TCIA API, retrieves metadata for the NSCLC-Radiomics collection, 
    filters for patients with both CT and RTSTRUCT modalities,
    and downloads the DICOM series for a subset of patients into a specified directory.

    Settings (from settings.py):
        RAW_DATA_PATH (Path): Directory where raw DICOM data will be saved.
        COLLECTION_NAME (str): Name of the TCIA collection to download.
    
    Args:
        n_patients_to_download (int, optional): Number of valid patients to retrieve. 
            Defaults to N_PATIENTS from settings.py.

    Raises:
        ValueError: it raises an error if no patients with both CT and RTSTRUCT modalities are found in the specified collection.
    """
    # Creation of folder for raw data
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Metadata retrieval
    logger.info("Connecting to TCIA... Retrieving series list.")
    df = nbia.getSeries(collection=COLLECTION_NAME, format="df")

    # 2. Filtering logic: we are looking for patients with CT + RTSTRUCT
    patients_ct = set(df[df[modality] == CT][patientID])
    patients_rt = set(df[df[modality] == RTSTRUCT][patientID])
    valid_patients = sorted(list(patients_ct.intersection(patients_rt)))

    if len(valid_patients) == 0:
        raise ValueError(f"Error: No patients found with both {CT} and {RTSTRUCT} in '{COLLECTION_NAME}'.")

    logger.info(f"Found {len(valid_patients)} complete patients.")

    # 3. Select a subset (e.g. 100)
    if len(valid_patients) < n_patients_to_download:
        # If n_patients_to_download is greater than the available patients
        logger.warning(f"Warning: Requested {n_patients_to_download} patients, but only {len(valid_patients)} are available.")
        actual_download_count = len(valid_patients)
    else:
        actual_download_count = n_patients_to_download

    subset_patients = valid_patients[:actual_download_count]
    df_to_download = df[df[patientID].isin(subset_patients)]

    series_dict_list = df_to_download.to_dict(orient='records')

    logger.info(f"Starting download for {actual_download_count} patients in {RAW_DATA_PATH}...")
    nbia.downloadSeries(series_dict_list, path=RAW_DATA_PATH)
    logger.info("Download completed successfully!")

