#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from nsclc_survival._download_data import download_nsclc_radiomics_data
from nsclc_survival.settings import patientID, modality, CT, RTSTRUCT

def test_download_nsclc_radiomics_data_success(mocker):    
    """
    Test that the download filtering logic correctly identifies patients 
    who have both CT and RTSTRUCT modalities.
    """
    # 1. Mock tcia_utils functions to block the network calls
    mock_get_series = mocker.patch('tcia_utils.nbia.getSeries')
    mock_download = mocker.patch('tcia_utils.nbia.downloadSeries')      # to avoid downloading
    mocker.patch('pathlib.Path.mkdir')
    
    # 2. Create a fake DataFrame of metadata 
    fake_metadata = pd.DataFrame([
        {patientID: "PAT_001", modality: CT, "SeriesInstanceUID": "1.1"},
        {patientID: "PAT_001", modality: RTSTRUCT, "SeriesInstanceUID": "1.2"}, # Complete
        {patientID: "PAT_002", modality: CT, "SeriesInstanceUID": "2.1"},       # Only CT (not valid)
        {patientID: "PAT_003", modality: RTSTRUCT, "SeriesInstanceUID": "3.1"}  # Only RT (not valid)
    ])
    mock_get_series.return_value = fake_metadata
    
    # 3. Execute the filtering logic
    download_nsclc_radiomics_data()

    assert mock_download.called

    called_args = mock_download.call_args[0][0]
    patient_ids_in_download = [record[patientID] for record in called_args]

    # --- Verify the ValueError check wasn't triggered ---
    # In this test case valid_patients is NOT empty, so len() == 0 should be false
    assert len(patient_ids_in_download) > 0
    
    # 4. Assert: the only valid patient should be PAT_001
    assert "PAT_001" in patient_ids_in_download
    assert "PAT_002" not in patient_ids_in_download
    assert "PAT_003" not in patient_ids_in_download

    # --- Verify the subset logic ---
    unique_patients_downloaded = len(set(patient_ids_in_download))
    assert unique_patients_downloaded == 1  # Since we only have 1 valid patient in mock data

def test_download_nsclc_radiomics_data_raises_value_error(mocker):
    """
    Test that download_nsclc_radiomics_data raises a ValueError if no patients match 
    both CT and RTSTRUCT modalities.
    """
    mock_get_series = mocker.patch('tcia_utils.nbia.getSeries')
    mocker.patch('tcia_utils.nbia.downloadSeries')
    mocker.patch('pathlib.Path.mkdir')
    
    # Create empty metadata or data that doesn't intersect
    fake_metadata = pd.DataFrame([
        {patientID: "PAT_002", modality: CT, "SeriesInstanceUID": "2.1"},
        {patientID: "PAT_003", modality: RTSTRUCT, "SeriesInstanceUID": "3.1"}
    ])
    mock_get_series.return_value = fake_metadata
    
    # Verify that the logic accurately raises the ValueError 
    with pytest.raises(ValueError) as exc_info:
        download_nsclc_radiomics_data()
    
    assert "no patients found" in str(exc_info.value).lower()
    