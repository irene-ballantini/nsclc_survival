import pytest
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from ruamel.yaml import YAML
from nsclc_survival.feature_extractor import FeatureExtractor

@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Fixture that creates a temporary folder structure with dummy data for testing.
    Creates an empty configuration folder and two test patients.

    Args:
        tmp_path (pathlib.Path): Pytest fixture providing a temporary directory unique 
            to the test invocation.

    Returns:
        dict: A dictionary containing paths to the configuration file and the preprocessed data directory.
    """
    # 1. Create a fake configuration file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "pyradiomics_config.yaml"

    # Configurate some parameters for PyRadiomics
    config_data = {
        "setting": {          
            "binWidth": 25
        }
    }
    
    ryaml = YAML()
    
    with open(config_file, "w") as f:
        ryaml.dump(config_data, f)

    # 2. Create the folder of preprocessed data
    data_dir = tmp_path / "preprocessed"
    data_dir.mkdir()

    # Prepare a fake 3D image (5x5x5) and a fake binary mask
    # Image with random values, mask with a central cube of 1s
    img_data = np.random.randint(0, 100, size=(5, 5, 5), dtype=np.int16)
    mask_data = np.zeros((5, 5, 5), dtype=np.int8)
    mask_data[2:4, 2:4, 2:4] = 1  # A small internal target to avoid empty segmentation errors

    sitk_img = sitk.GetImageFromArray(img_data)
    sitk_mask = sitk.GetImageFromArray(mask_data)

    # Create two valid patients
    for p_id in ["LUNG1-001", "LUNG1-002"]:
        p_dir = data_dir / p_id
        p_dir.mkdir()
        sitk.WriteImage(sitk_img, str(p_dir / "image.nii.gz"))
        sitk.WriteImage(sitk_mask, str(p_dir / "label.nii.gz"))

    # Create an "incomplete" patient (missing the mask) to test the SKIP functionality
    bad_p_dir = data_dir / "LUNG1-003_incomplete"
    bad_p_dir.mkdir()
    sitk.WriteImage(sitk_img, str(bad_p_dir / "image.nii.gz"))

    return {
        "config_path": config_file,
        "preprocessed_path": data_dir
    }


def test_extractor_initialization(dummy_dataset):
    """
    Verify the extractor initializes correctly with or without a config file.

    Args:
        dummy_dataset (dict): Pytest fixture that provides the paths to the simulated data.
            Contains the keys:
            - 'config_path': Path to the fake pyradiomics_config.yaml file.
            - 'preprocessed_path': Path to the folder containing 2 valid patients 
              (LUNG1-001, LUNG1-002) and 1 incomplete.
    """
    # Test with config file present
    extractor = FeatureExtractor(dummy_dataset["config_path"])
    assert extractor.config_path.exists(), "The path to the configuration file stored does not exist on disk!"
    assert extractor.extractor is not None, "Radiomics extractor was not initialized"
    assert extractor.extractor.settings['binWidth'] == 25, "The binWidth setting from the config file was not correctly loaded."

    # Test with config file missing 
    fake_path = Path("path/non_existing/config.yaml")
    extractor_default = FeatureExtractor(fake_path)
    assert extractor_default.extractor is not None, "Failed to initialize the radiomics extractor with default settings"


def test_extract_all_features_success(dummy_dataset):
    """
    Verify that feature extraction succeeds for valid patients.
    
    Args:
        dummy_dataset (dict): Pytest fixture that provides the paths to the simulated data.
            Contains the keys:
            - 'config_path': Path to the fake pyradiomics_config.yaml file.
            - 'preprocessed_path': Path to the folder containing 2 valid patients 
              (LUNG1-001, LUNG1-002) and 1 incomplete.
    """
    extractor = FeatureExtractor(dummy_dataset["config_path"])
    results = extractor.extract_all_features(dummy_dataset["preprocessed_path"])

    # Just 2 results expected since only 2 valid patients are present (the third was incomplete and should be skipped)
    assert len(results) == 2

    # Verify the structure of the output for the first patient
    first_patient_feats = results[0]
    assert isinstance(first_patient_feats, dict)    # Check that the output is a dictionary
    assert "PatientID" in first_patient_feats, "The output dictionary should contain the 'PatientID' key"
    assert first_patient_feats["PatientID"] in ["LUNG1-001", "LUNG1-002"], "The PatientID should match one of the valid patients"

def test_extract_radiomics_single_patient(dummy_dataset):
    """
    Verify the targeted extraction for a single patient.

    Args:
        dummy_dataset (dict): Pytest fixture that provides the paths to the simulated data.
            Contains the keys:
            - 'config_path': Path to the fake pyradiomics_config.yaml file.
            - 'preprocessed_path': Path to the folder containing 2 valid patients 
              (LUNG1-001, LUNG1-002) and 1 incomplete.
    """
    extractor = FeatureExtractor(dummy_dataset["config_path"])
    
    p1_dir = dummy_dataset["preprocessed_path"] / "LUNG1-001"
    image_path = p1_dir / "image.nii.gz"
    mask_path = p1_dir / "label.nii.gz"

    feat_dict = extractor.extract_radiomics(image_path, mask_path, "LUNG1-001")

    assert isinstance(feat_dict, dict), "The output of extract_radiomics should be a dictionary"
    assert feat_dict["PatientID"] == "LUNG1-001", "The PatientID in the output dictionary should match the input patient ID"

    # Verify that the diagnostic metadata has been effectively removed
    for key in feat_dict.keys():
        assert not key.startswith("diagnostics_"), "Diagnostic metadata should be removed from the output dictionary"

    # PyRadiomics estract always at least a dozen of shape/intensity features
    assert len(feat_dict) > 5 , "The output dictionary should contain at least 5 features"
    
    # Check that the expected feature is present in the output dictionary
    assert "original_shape_VoxelVolume" in feat_dict

    # Check calculated values are actually numbers (and not None or strings)
    assert isinstance(feat_dict["original_shape_VoxelVolume"], (int, float))

    # Verify that the calculated volume for our fake cube is greater than zero
    assert feat_dict["original_shape_VoxelVolume"] > 0