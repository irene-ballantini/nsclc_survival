import pytest
import pydicom
from pathlib import Path
# Import the function
from nsclc_survival._organize_data import organize_dicom_data

def test_organize_dicom_data_success(mocker, tmp_path):
    """    
    Test that organize_dicom_data correctly reorganizes DICOM files 
    into PatientID/Modality folders and removes the raw folder.
    
    Args:
        mocker (pytest_mock.plugin.MockerFixture): Pytest mocker fixture for patching.
        tmp_path (pathlib.Path): Temporary directory path provided by pytest.
    """
    # 1. SETUP: Create the fake structure inside the temporary folder tmp_path
    fake_raw = tmp_path / "raw"
    fake_organized = tmp_path / "organized"
    
    fake_ct_series = fake_raw / "series_ct"
    fake_rt_series = fake_raw / "series_rt"

    fake_ct_series.mkdir(parents=True)
    fake_rt_series.mkdir(parents=True)

    # Create two fake files .dcm
    file_ct = fake_ct_series / "ct.dcm"
    file_rt = fake_rt_series / "rt.dcm"
    file_ct.write_text("dummy dicom content ct")
    file_rt.write_text("dummy dicom content rt")
    
    # 2. MOCK: Simulate the reading of DICOM metadata
    def mock_dcmread_by_folder(filepath):
        mock_data = mocker.MagicMock()
        mock_data.PatientID = "PAT_999"
        # Recognizes the modality from the name of the file
        if "ct" in str(filepath).lower():
            mock_data.Modality = "CT"
        else:
            mock_data.Modality = "RTSTRUCT"
        return mock_data

    mocker.patch("pydicom.dcmread", side_effect=mock_dcmread_by_folder)
    
    # 3. EXECUTION: Call the function
    organize_dicom_data(fake_raw, fake_organized)
    
    # 4. ASSERTIONS: Verify the result
    expected_ct_destination = fake_organized / "PAT_999" / "CT"
    expected_rt_destination = fake_organized / "PAT_999" / "RTSTRUCT"
    
    # Verify that the folder exists and contains the two file
    assert expected_ct_destination.exists()
    assert expected_rt_destination.exists()
    assert (expected_ct_destination / "ct.dcm").exists()
    assert (expected_rt_destination / "rt.dcm").exists()
    
    # Verify that the old raw folder has been removed
    assert not fake_raw.exists()

def test_organize_dicom_data_raw_not_found(caplog, tmp_path):
    """
    If folder doesn't exist returns a Warning.

    Args:
        caplog (pytest.LogCaptureFixture): Pytest fixture for capturing log messages.
        tmp_path (pathlib.Path): Temporary directory path provided by pytest.
    """
    fake_raw = tmp_path / "non_existing_raw"
    fake_organized = tmp_path / "organized"

    # Execution 
    organize_dicom_data(fake_raw, fake_organized)

    # Verify that it prints the correct Warning using the pytest fixture 'capsys'
    assert any("Warning: raw folder not found" in record.message for record in caplog.records)
    assert any(record.levelname == "WARNING" for record in caplog.records)

def test_organize_dicom_data_corrupted_file_keeps_folder(mocker, tmp_path):
    """
    If dcmread fails, the original raw folder is not removed.

    Args:
        mocker (pytest_mock.plugin.MockerFixture): Pytest mocker fixture for patching.
        tmp_path (pathlib.Path): Temporary directory path provided by pytest.
    """
    fake_raw = tmp_path / "raw"
    fake_organized = tmp_path / "organized"
    
    corrupted_folder = fake_raw / "series_corrupted"
    corrupted_folder.mkdir(parents=True)
    
    file_bad = corrupted_folder / "bad.dcm"
    file_bad.write_text("corrupted data")

    # MOCK: Simulate that dcmread raises an error
    mocker.patch("pydicom.dcmread", side_effect=Exception("Invalid DICOM file"))

    # EXECUTION
    organize_dicom_data(fake_raw, fake_organized)

    # ASSERTIONS
    assert corrupted_folder.exists()
    assert fake_raw.exists()