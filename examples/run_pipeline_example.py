from pathlib import Path
import logging
import pandas as pd
import shutil

# Import of the classes and settings of the package
from nsclc_survival import (
    RadiomicsPreprocessor, 
    FeatureExtractor, 
    RadiomicsClinicalDataProcessor, 
    LassoCoxModel,
    DeepCoxModel, 
    SurvivalRiskClassifier,
    settings,
    utils,
    _download_data,
    _organize_data
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)

    # 1. DOWNLOAD and DATA ORGANIZATION
    # N_PATIENTS is 100 by default in settings.py 

    # NOTE: You can change N_PATIENTS for a fast test.
    # If you change this number and you already have data in the 'data/' subfolders,
    # because you have already run the script, please manually clear them
    # or let the automatic cleanup below reset the cache for you,
    # or, if you don't want to delete any file, change the 'DATA_DIR' constant
    # in settings.py to a different folder and run the script again 
    # (e.g., by setting DATA_DIR = BASE_DIR / "data_new").

    # To do a fast test, you can uncomment the line below.
    # settings.N_PATIENTS = 20      # the train set cannot be smaller than 15 if you want to run the models

    # --- Automatic cache reset if mismatch is detected ---
    if settings.ORGANIZED_DATA_PATH.exists():
        current_on_disk = len([f for f in settings.ORGANIZED_DATA_PATH.iterdir() if f.is_dir()])
        if current_on_disk != settings.N_PATIENTS:
            logger.info("[WARNING] Patient count mismatch detected. Automatically resetting data cache...")
            for p in [settings.ORGANIZED_DATA_PATH, settings.PREPROCESSED_DATA_PATH, settings.RAD_FEATURES_CSV_PATH]:
                if p.exists():
                    if p.is_dir(): shutil.rmtree(p)
                    else: p.unlink()

    logger.info(f"[STEP 1] Download and organization of data for {settings.N_PATIENTS} patients...")
    _download_data.download_nsclc_radiomics_data(n_patients_to_download=settings.N_PATIENTS)
    _organize_data.organize_dicom_data(raw_path=settings.RAW_DATA_PATH, organized_path=settings.ORGANIZED_DATA_PATH)

    logger.info("=" * 80)
    # 2. PREPROCESSING and FEATURE EXTRACTION
    logger.info("[STEP 2] Image preprocessing and radiomic feature extraction...")
    processor = RadiomicsPreprocessor(
        organized_path=settings.ORGANIZED_DATA_PATH, 
        preprocessed_path=settings.PREPROCESSED_DATA_PATH
    )    
    processor.process_all_patients()

    fe = FeatureExtractor(config_path=settings.RADIOMICS_CONFIG_PATH)
    extracted_features = fe.extract_all_features(preprocessed_path=settings.PREPROCESSED_DATA_PATH)
    
    if extracted_features:
        utils.save_features_to_csv(features_list=extracted_features, output_path=settings.RAD_FEATURES_CSV_PATH)
        logger.info(f"[SUCCESS] Radiomic features saved to: {settings.RAD_FEATURES_CSV_PATH}")
    else:
        logger.error("[ERROR] Feature extraction failed.")
        return

    logger.info("=" * 80)
    # 3. MERGING WITH CLINICAL DATA AND DATASET SPLITTING
    logger.info("[STEP 3] Loading clinical data and splitting Train/Test...")
    data_processor = RadiomicsClinicalDataProcessor(
        radiomics_path=settings.RAD_FEATURES_CSV_PATH, 
        clinical_path=settings.CLINICAL_FEATURES_CSV_PATH
    )
    
    df_merged = data_processor.load_and_merge(
        settings.patientID, settings.stage_col, settings.gender_col, 
        settings.histology_col, settings.stage_mapping, settings.gender_mapping
    )
    
    X_total, y_total, X_train, X_test, y_train, y_test = data_processor.split_and_standardize(
        patientID=settings.patientID,
        survival_time_col=settings.survival_time_col,
        event_status_col=settings.event_status_col,
        train_size=0.8,
        random_seed=42
    )

    logger.info("=" * 80)
    # 4. MODEL 1: LASSO-COX
    logger.info("[STEP 4] Training and evaluation of Lasso-Cox...")
    lasso_cox = LassoCoxModel(
        feature_names=data_processor.feature_names,
        stage=settings.stage_col,
        gender=settings.gender_col,
        histology=settings.histology_col
    )
    lasso_cox.fit_crossval(X_train, y_train, X_total, y_total, cv=5)
    
    c_index_cox = lasso_cox.evaluate_model(X_test, y_test)
    logger.info(f"-> Lasso-Cox Test C-Index: {c_index_cox:.4f}")

    logger.info("=" * 80)
    # 5. MODEL 2: DEEP COX
    logger.info("[STEP 5] Training and evaluation of Deep Cox...")
    deep_cox = DeepCoxModel(
        input_dim=X_train.shape[1], 
        hidden_dims=[64, 32], 
        dropout_rate=0.1, 
        lr=5e-4, 
        weight_decay=1e-4
    )  
    deep_cox.fit(X_train, y_train, epochs=100, batch_size=32)
    
    deep_risk_scores = deep_cox.compute_risk_scores(X_input=X_test)
    c_index_deep = deep_cox.evaluate_model(X_test, y_test, risk_scores=deep_risk_scores)
    logger.info(f"-> Deep Cox Test C-Index: {c_index_deep:.4f}")
    
    logger.info("\n[SUCCESS] Example Pipeline executed successfully!")

if __name__ == "__main__":
    main()