#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from unittest.mock import MagicMock

# Import the class
from nsclc_survival.preprocessing import RadiomicsPreprocessor  

@pytest.fixture
def setup_mock_dataset(tmp_path):
    """
    Fixture to create a simulated DICOM/RTSTRUCT dataset structure.

    Generates dummy .dcm files inside a structured patient directory to mimic
    an organized clinical dataset.

    Args:
        tmp_path (pathlib.Path): Pytest fixture providing a temporary directory unique 
            to the test invocation.

    Returns:
        tuple: A tuple containing:
            - organized_dir (pathlib.Path): Path to the simulated raw DICOM directory.
            - preprocessed_dir (pathlib.Path): Path to the output directory for processed files.
            - patient_id (str): The ID of the simulated patient ("PAT_001").
    """
    organized_dir = tmp_path / "organized"
    preprocessed_dir = tmp_path / "preprocessed"
    
    organized_dir.mkdir()
    preprocessed_dir.mkdir()
    
    # Create the structure for one patient
    patient_id = "PAT_001"
    patient_dir = organized_dir / patient_id
    ct_dir = patient_dir / "CT"
    rt_dir = patient_dir / "RTSTRUCT"
    
    ct_dir.mkdir(parents=True)
    rt_dir.mkdir(parents=True)
    
    # Create fake files 
    (rt_dir / "rt_mock.dcm").touch()
    (ct_dir / "ct_slice1.dcm").touch()
    
    return organized_dir, preprocessed_dir, patient_id

@pytest.mark.parametrize("input_spacing, expected_size", [
    ([0.8, 0.8, 2.0], (8, 8, 10)),   # Case 1: Image to resample (New Size = Old Size * Input Spacing / Target Spacing)
    ([1.0, 1.0, 1.0], (10, 10, 5))   # Case 2: Image already isotropic (It remains the same)
])
def test_pipeline_success(input_spacing, expected_size, setup_mock_dataset, mocker):
    """
    Test that the preprocessor successfully completes the extraction and the resampling.

    Verify that given a valid CT series and a matching RTSTRUCT with the target ROI,
    the preprocessor outputs NIfTI images (.nii.gz) for both image and mask, and 
    correctly resamples them to isotropic (1.0mm) spacing. Handles both anisotropic 
    and native isotropic input images.

    Args:
        input_spacing (list): The pixel spacing of the simulated input CT image.
        expected_size (tuple): The expected pixel dimensions of the output NIfTI image.
        setup_mock_dataset (tuple): Tuple containing (organized_dir, preprocessed_dir, patient_id)
            provided by the setup fixture.
        mocker (pytest_mock.plugin.MockerFixture): Pytest-mock fixture to patch internal objects.
    """
    organized_dir, preprocessed_dir, patient_id = setup_mock_dataset

    # Create the mocks using the native fixture 'mocker' of pytest
    mock_get_dicoms = mocker.patch('nsclc_survival.preprocessing.sitk.ImageSeriesReader.GetGDCMSeriesFileNames')
    mock_sitk_execute = mocker.patch('nsclc_survival.preprocessing.sitk.ImageSeriesReader.Execute')
    mock_rt_builder = mocker.patch('nsclc_survival.preprocessing.RTStructBuilder.create_from')
    
    # 1. Mocking of SimpleITK for reading the series CT
    # Create a fake image ITK (e.g. 10x10x5 voxel, spacing 0.8x0.8x2.0mm)
    fake_ct = sitk.Image(10, 10, 5, sitk.sitkInt16)
    fake_ct.SetSpacing(input_spacing)
    fake_ct.SetOrigin([0.0, 0.0, 0.0])
    mock_sitk_execute.return_value = fake_ct
    mock_get_dicoms.return_value = ["fake_ct_slice1.dcm"]

    # 2. Mocking of RTStructBuilder and of the returned rtstruct object
    mock_rtstruct_instance = MagicMock()
    # Communicate that the "GTV-1" ROI is present
    mock_rtstruct_instance.get_roi_names.return_value = ["GTV-1", "SpinalCord"]
    # Return a numpy mask coherent with the dimension of the CT (H, W, Slices) -> (10, 10, 5)
    fake_mask_np = np.zeros((10, 10, 5), dtype=bool)
    fake_mask_np[3:7, 3:7, 2:4] = True  # Put a fake cube as tumor
    mock_rtstruct_instance.get_roi_mask_by_name.return_value = fake_mask_np
    
    mock_rt_builder.return_value = mock_rtstruct_instance

    # 3. Execution of the Preprocessor
    preprocessor = RadiomicsPreprocessor(organized_dir, preprocessed_dir)
    preprocessor.process_all_patients()

    # 4. Assertions
    output_patient_dir = preprocessed_dir / patient_id
    assert output_patient_dir.exists(), "The patient output folder has not been created."
    
    image_nii = output_patient_dir / "image.nii.gz"
    label_nii = output_patient_dir / "label.nii.gz"
    
    assert image_nii.exists(), "The file image.nii.gz doesn't exist."
    assert label_nii.exists(), "the file label.nii.gz doesn't exist."

    # Verify that the isotropic resampling (1mm x 1mm x 1mm) was correctly done
    saved_img = sitk.ReadImage(str(image_nii))
    saved_lbl = sitk.ReadImage(str(label_nii))
    
    assert saved_img.GetSpacing() == (1.0, 1.0, 1.0)
    assert saved_lbl.GetSpacing() == (1.0, 1.0, 1.0)
    assert saved_img.GetSize() == expected_size     # (10*0.8/1.0 = 8, 10*0.8/1.0 = 8, 5*2.0/1.0 = 10)
    assert saved_img.GetSize() == saved_lbl.GetSize(), "CT and Mask don't have the same dimension!"

def test_pipeline_missing_roi(setup_mock_dataset, mocker):
    """
    Test that the patient is skipped if the target ROI (GTV-1) doesn't exist
    
    Ensures that when the RTSTRUCT file contains non-target ROIs (e.g., Heart, Lung),
    the pipeline skips processing for that patient and does not write any output files.

    Args:
        setup_mock_dataset (tuple): Tuple containing (organized_dir, preprocessed_dir, patient_id).
        mocker (pytest_mock.plugin.MockerFixture): Pytest-mock fixture to patch internal objects.
    """
    organized_dir, preprocessed_dir, patient_id = setup_mock_dataset

    mock_rt_builder = mocker.patch('nsclc_survival.preprocessing.RTStructBuilder.create_from')
    
    # Mocking of an rtstruct WITHOUT GTV-1
    mock_rtstruct_instance = MagicMock()
    mock_rtstruct_instance.get_roi_names.return_value = ["Heart", "Lung_L"]
    mock_rt_builder.return_value = mock_rtstruct_instance

    preprocessor = RadiomicsPreprocessor(organized_dir, preprocessed_dir)
    preprocessor.process_all_patients()

    # Verify nothing has been saved
    output_patient_dir = preprocessed_dir / patient_id / "label.nii.gz"
    assert not output_patient_dir.exists(), "The file shouldn't have been created because the ROI was missing."

