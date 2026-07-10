#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import time
import logging
from radiomics import featureextractor

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """ 
    Class to extract radiomics features from preprocessed images and masks using PyRadiomics.

    Attributes:
        config_path (Path): Path to the pyradiomics_config.yaml file.
        extractor (RadiomicsFeatureExtractor): An instance of the PyRadiomics feature extractor.
    """
    def __init__(self, config_path):

        self.config_path = Path(config_path)

        radiomics_logger = logging.getLogger("radiomics")
        radiomics_logger.setLevel(logging.ERROR)

        if self.config_path.exists():
            self.extractor = featureextractor.RadiomicsFeatureExtractor(str(self.config_path))
        else:
            logger.warning(f"[WARNING] Config file not found in {self.config_path}. Using default settings.")
            self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        logger.info(f"Image types enabled: {self.extractor.enabledImagetypes}")
        logger.info(f"Feature classes enabled: {self.extractor.enabledFeatures}")   

    def extract_all_features(self, preprocessed_path):
        """
        Extract features from all patients in the directory.

        Args:
            preprocessed_path (Path): Path to the directory containing preprocessed patient folders.

        Returns:
            list of dict: A list where each element is a dictionary of extracted features 
                for a single patient. 

                Each dictionary contains:
                    - "PatientID" (str): The unique identifier of the patient.
                    - Pyradiomics feature names (str): The calculated radiomics 
                      features with their corresponding values
        """
        preprocessed_path = Path(preprocessed_path)
        patient_folders = sorted([f for f in preprocessed_path.iterdir() if f.is_dir()])
        logger.info(f"Found {len(patient_folders)} patients.")
        
        start_total_time = time.time()    # Start total timer
        all_features = []
        for p in patient_folders:
            image_path = p / "image.nii.gz"
            mask_path = p / "label.nii.gz"
            patient_id = p.name

            if not image_path.exists() or not mask_path.exists():
                logger.info(f"[SKIP] {patient_id}: image or mask missing")
                continue

            logger.info(f"Feature extraction for {patient_id}...")
            start_patient_time = time.time()  # Start timer for this patient
            
            try:
                feats = self.extract_radiomics(image_path, mask_path, patient_id)
                all_features.append(feats)
                
                # Calculate duration time for this patient 
                patient_duration = time.time() - start_patient_time
                logger.info(f"--> Done in {patient_duration:.2f} seconds.\n")

            except Exception as e:
                logger.error(f"[ERROR] {patient_id}: {e}")
                continue
        
        # Calculate total duration time
        total_duration = time.time() - start_total_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = total_duration % 60

        logger.info(f"\nExtraction completed.")
        logger.info(f"Number of processed patients: {len(all_features)}")
        logger.info(f"Number of  extracted radiomics features (columns): {len(all_features[0]) - 1 if all_features else 0}")  # Subtract 1 for PatientID column

        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds:.2f}s"
        else:
            time_str = f"{minutes}m {seconds:.2f}s"
            
        avg_time = total_duration / len(all_features) if all_features else 0.00
        logger.info(f"Total execution time: {time_str} (Average: {avg_time:.2f}s per patient)")

        return all_features

    def extract_radiomics(self, image_path, mask_path, patient_id):
        """
        Extract features from the single patient, removing metadata diagnostics.

        Args:
            image_path (Path): Path to the image file
            mask_path (Path): Path to the mask file
            patient_id (str): ID of the patient

        Returns:
            dict: A dictionary containing the extracted feature names (keys) 
                  and their corresponding calculated values (values).
        """
        result = self.extractor.execute(str(image_path), str(mask_path))
        
        # Remove metadata diagnostics
        clean = {}
        for k, v in result.items():
            if k.startswith("diagnostics_"):
                continue
            clean[k] = v

        clean["PatientID"] = patient_id
        return clean
