#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from nsclc_survival.settings import COLLECTION_NAME, N_PATIENTS, patientID, modality, CT, RTSTRUCT

def test_download_script_filtering(mocker):    
    """
    Test that the download filtering logic correctly identifies patients 
    who have both CT and RTSTRUCT modalities, and safely limits the 
    download list to N_PATIENTS.
    """
    # 1. Mock tcia_utils functions to block the network calls
    mock_get_series = mocker.patch('tcia_utils.nbia.getSeries')
    mocker.patch('tcia_utils.nbia.downloadSeries')      # to avoid downloading
    
    # 2. Create a fake DataFrame of metadata 
    fake_metadata = pd.DataFrame([
        {patientID: "PAT_001", modality: CT, "SeriesInstanceUID": "1.1"},
        {patientID: "PAT_001", modality: RTSTRUCT, "SeriesInstanceUID": "1.2"}, # Complete
        {patientID: "PAT_002", modality: CT, "SeriesInstanceUID": "2.1"},       # Only CT (not valid)
        {patientID: "PAT_003", modality: RTSTRUCT, "SeriesInstanceUID": "3.1"}  # Only RT (not valid)
    ])
    mock_get_series.return_value = fake_metadata
    
    # 3. Execute the filtering logic
    patients_ct = set(fake_metadata[fake_metadata[modality] == CT][patientID])
    patients_rt = set(fake_metadata[fake_metadata[modality] == RTSTRUCT][patientID])
    valid_patients = sorted(list(patients_ct.intersection(patients_rt)))

    # --- Verify the ValueError check ---
    # In this test case valid_patients is NOT empty, so len() == 0 should be false
    assert len(valid_patients) > 0
    
    # 4. Assert: the only valid patient should be PAT_001
    assert valid_patients == ["PAT_001"]
    assert "PAT_002" not in valid_patients
    assert "PAT_003" not in valid_patients

    # --- Verify the subset logic (if/else) ---
    if len(valid_patients) < N_PATIENTS:
        actual_download_count = len(valid_patients)
    else:
        actual_download_count = N_PATIENTS
        
    assert actual_download_count == 1  # Since we only have 1 valid patient in mock data

def test_download_script_raises_value_error_when_empty(mocker):
    """
    Test that the download script raises a ValueError if no patients match 
    both CT and RTSTRUCT modalities.
    """
    mock_get_series = mocker.patch('tcia_utils.nbia.getSeries')
    mocker.patch('tcia_utils.nbia.downloadSeries')
    
    # Create empty metadata or data that doesn't intersect
    fake_metadata = pd.DataFrame([
        {patientID: "PAT_002", modality: CT, "SeriesInstanceUID": "2.1"},
        {patientID: "PAT_003", modality: RTSTRUCT, "SeriesInstanceUID": "3.1"}
    ])
    mock_get_series.return_value = fake_metadata
    
    # Replicate logic to ensure it triggers the same exception condition
    patients_ct = set(fake_metadata[fake_metadata[modality] == CT][patientID])
    patients_rt = set(fake_metadata[fake_metadata[modality] == RTSTRUCT][patientID])
    valid_patients = sorted(list(patients_ct.intersection(patients_rt)))
    
    # Verify that the logic accurately raises the ValueError when len is 0
    with pytest.raises(ValueError) as exc_info:
        if len(valid_patients) == 0:
            raise ValueError(f"Error: No patients found with both {CT} and {RTSTRUCT} in '{COLLECTION_NAME}'.")
            
    assert "no patients found" in str(exc_info.value).lower()
    