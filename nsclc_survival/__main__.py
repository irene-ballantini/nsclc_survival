#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import pandas as pd
import shutil

from nsclc_survival import ( 
    __version__, 
    RadiomicsPreprocessor, 
    FeatureExtractor, 
    RadiomicsClinicalDataProcessor, 
    LassoCoxModel,
    DeepCoxModel, 
    SurvivalRiskClassifier
    )

from nsclc_survival import _download_data
from nsclc_survival import _organize_data
from nsclc_survival import utils
from nsclc_survival import settings

logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s', 
    force=True
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments for the nsclc_survival package.
    This function sets up the argument parser, defines the expected arguments,
    and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments as a Namespace object.

    Raises:
        argparse.ArgumentError: If there is an error in the argument parsing, such as a missing required
        argument or an invalid value.
    """
    # Global information
    parser = argparse.ArgumentParser(
        prog='nsclc_survival',
        argument_default=None,
        add_help=True,
        prefix_chars='-',
        allow_abbrev=True,
        exit_on_error=True,
        description='NSCLC Survival Analysis Pipeline: Survival Time prediction using CT-extracted features and clinical data.'
    )
    
    # nsclc_survival --n-patients <int>
    # This option allows the user to specify the number of patients to download.
    # The default value is 100 and is provided in the separated settings.py file, 
    # which is possible to modify to customize some parameters of the pipeline execution. 
    # The number of patients to download can be modified both via command line and in the settings file.
    parser.add_argument(
        "--n-patients", 
        type=int, 
        default=settings.N_PATIENTS,    # default set to 100 in settings.py 
        help=f"Number of patients to download (default from settings: {settings.N_PATIENTS})"
    )   
    
    # nsclc_survival --cv-folds <int>
    # This option allows the user to specify the number of folds for the Lasso Cross-Validation
    # in Cox model.
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=5, 
        help="Folds for the Lasso Cross-Validation in Cox model. Default to 5."
    )
    
    # nsclc_survival --epochs <int>
    # This option allows the user to specify the number of epochs of training for Deep Cox
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=200, 
        help="Epochs of training for Deep Cox"
    )
    
    # nsclc_survival --batch-size <int> 
    # This option allows the user to specify the batch size of training for Deep Cox
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64, 
        help="Batch size of training for Deep Cox"
    )
    
    # nsclc_survival --hidden-dims <int>...
    # This option allows the user to specify the architecture of the hidden layers for Deep Cox 
    parser.add_argument(
        "--hidden-dims", 
        type=int, 
        nargs="+", 
        default=[64, 32], 
        help="Hidden dimensions for the Deep Cox neural network (e.g., --hidden-dims 128 64 32)"
    )

    # nsclc_survival --version
    parser.add_argument(
        "-v", "--version", 
        action="version", 
        version=f"%(prog)s {__version__}", 
        help="Show the current package version and exit"
    ) 

    # nsclc_survival --skip-download
    # Flag to skip the download and organization of DICOM data (use local data if present)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download and organization of DICOM data (use local data if present)"
    )

    # nsclc_survival --skip-extraction
    # Flag to skip the extraction of radiomic features
    parser.add_argument(
        "--skip-extraction", 
        action="store_true", 
        help="Skip the preprocessing and extraction of radiomic features"
    )

    return parser.parse_args()

def main (): 
    args = parse_args()
    logger.info(f"Package version: {__version__}")

    # Check how many patients are currently on disk
    if settings.ORGANIZED_DATA_PATH.exists():
        current_patients_on_disk = len(sorted([f for f in settings.ORGANIZED_DATA_PATH.iterdir() if f.is_dir()]))
    else:
        current_patients_on_disk = 0

    # Data are ready only if the folder is not empty 
    # and the number of patients on disk matches the requested number
    data_exists = (current_patients_on_disk > 0) and (current_patients_on_disk == args.n_patients)

    # Download and organize data
    if not args.skip_download and not data_exists:
        # Handling the mismatch before starting the download process
        if current_patients_on_disk > 0 and current_patients_on_disk != args.n_patients:
            logger.info(f"Patient count mismatch: Requested {args.n_patients}, but found {current_patients_on_disk} on disk.")
            logger.info(f"Cleaning up old organized data in {settings.ORGANIZED_DATA_PATH}...")
            
            # Empty the organized folder to avoid mixing old and new data
            try:
                if settings.ORGANIZED_DATA_PATH.exists():
                    shutil.rmtree(settings.ORGANIZED_DATA_PATH)
                settings.ORGANIZED_DATA_PATH.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Critical: Could not clean organized folder due to: {e}")
                logger.error("Please close any files/viewers inside the data folder and restart the script.")
                return

            logger.info(f"Re-downloading and organizing data for the new requested amount of {args.n_patients} patients...")
        else:
            logger.info(f"Download and organization of DICOM data in progress for {args.n_patients} patients...")

        _download_data.download_nsclc_radiomics_data(n_patients_to_download=args.n_patients)
        _organize_data.organize_dicom_data(raw_path=settings.RAW_DATA_PATH, organized_path=settings.ORGANIZED_DATA_PATH)
        # Update the status after the download is completed successfully
        data_exists = True

    elif args.skip_download:
        logger.info("Download skipped by user via --skip-download.")
    else:
        logger.info("Data already available locally. Skipping download.")

    # --- Security Block ---
    # If data are not available (because the user skipped the download or the download failed)
    # and the script is about to start a phase that requires the data, it stops.
    if not data_exists and not args.skip_extraction:
        logger.error("Critical: No organized data found in settings.ORGANIZED_DATA_PATH!")
        logger.error("You used --skip-download but the folder is empty. Cannot proceed with radiomics.")
        return 
    
    # Check if features are ready and valid
    features_exist = data_exists and settings.RAD_FEATURES_CSV_PATH.exists()

    # --- Radiomics Processing and Feature Extraction ---
    if not args.skip_extraction and not features_exist:
        if settings.RAD_FEATURES_CSV_PATH.exists():
            logger.info(f"Removing outdated features file: {settings.RAD_FEATURES_CSV_PATH}")
            try:
                settings.RAD_FEATURES_CSV_PATH.unlink() # Cancella il singolo file CSV
            except Exception as e:
                logger.error(f"Critical: Could not remove outdated features CSV: {e}")
                return
        logger.info("\n" + "="*100)
        logger.info(" 1. RUNNING RADIOMICS PREPROCESSING ".center(100, " "))
        logger.info("="*100)

        processor = RadiomicsPreprocessor(
            organized_path=settings.ORGANIZED_DATA_PATH, 
            preprocessed_path=settings.PREPROCESSED_DATA_PATH
        )    

        processor.process_all_patients()
    
        logger.info("\n" + "=" * 100)
        logger.info(" 2. RUNNING FEATURE EXTRACTION  ".center(100, " "))
        logger.info("=" * 100)

        preprocessed_exists = settings.PREPROCESSED_DATA_PATH.exists() and any(settings.PREPROCESSED_DATA_PATH.iterdir())
        
        if preprocessed_exists:
            fe = FeatureExtractor(config_path=settings.RADIOMICS_CONFIG_PATH)
    
            extracted_features = fe.extract_all_features(preprocessed_path=settings.PREPROCESSED_DATA_PATH)
            
            if extracted_features:
                utils.save_features_to_csv(features_list=extracted_features, output_path=settings.RAD_FEATURES_CSV_PATH)
                logger.info(f"Features successfully extracted and saved to {settings.RAD_FEATURES_CSV_PATH}")
            else:
                logger.error("No features were extracted. CSV file was not created.")
        else:
            logger.error(f"Critical: No preprocessed NIfTI data found in {settings.PREPROCESSED_DATA_PATH}. Aborting extraction.")
            return
    else:
        skip_reason = "MANUAL SKIP VIA CLI" if args.skip_extraction else "FEATURES CSV ALREADY EXISTS AND IS VALID"
        logger.info("\n" + "="*100)
        logger.info(f" SKIPPING RADIOMICS PIPELINE ({skip_reason}) ".center(100, " "))
        logger.info("="*100)
    
    # --- SECURITY BLOCK FOR DATA ANALYSIS ---
    # If the code gets here and features_exist is False, it means the user used --skip-extraction
    # BUT the CSV file on disk is missing or belongs to an old dataset/number of patients!
    if not features_exist:
        logger.error("\n" + "!"*100)
        logger.error("CRITICAL ERROR: FEATURES CSV IS MISSING OR OUTDATED COMPARED TO CURRENT DICOM DATA!")
        logger.error("You requested to skip extraction, but the existing CSV does not match the current patients.")
        logger.error("To prevent the downstream analysis from using corrupted/wrong data, the script will now STOP.")
        logger.error("!"*100)
        return

    logger.info("\n" + "=" * 100)
    logger.info(" 3. MODELLING ".center(100, " "))
    logger.info("=" * 100)

    logger.info("--- LOADING AND PROCESSING DATA ---")
    data_processor = RadiomicsClinicalDataProcessor(
        radiomics_path=settings.RAD_FEATURES_CSV_PATH, 
        clinical_path=settings.CLINICAL_FEATURES_CSV_PATH
    )
    
    # Load and merge data
    df_merged = data_processor.load_and_merge(settings.patientID, settings.stage_col, settings.gender_col, 
                                              settings.histology_col, settings.stage_mapping, settings.gender_mapping)
    
    # Split and standardize data
    X_total, y_total, X_train, X_test, y_train, y_test = data_processor.split_and_standardize(
        patientID = settings.patientID,
        survival_time_col=settings.survival_time_col,
        event_status_col=settings.event_status_col,
        train_size=0.8,
        random_seed=42
    )

    logger.info("\n" + "=" * 100)
    logger.info(" MODEL 1: LASSO-COX MODEL".center(100, " "))
    logger.info("=" * 100)

    logger.info("\n--- COX STEP 1: TRAINING LASSO-COX MODEL WITH CROSS-VALIDATION ---")
    # Initialize the Lasso-Cox model
    lasso_cox = LassoCoxModel(
        feature_names=data_processor.feature_names,
        stage=settings.stage_col,
        gender=settings.gender_col,
        histology=settings.histology_col
    )
    
    # Execute Cross-Validation to find the optimal alpha and train the final model
    # You can change the number of folds by modifying cv (e.g., cv=5)
    lasso_cox.fit_crossval(X_train, y_train, X_total, y_total, cv=args.cv_folds)

    logger.info("\n--- COX STEP 2: EXTRACTING SELECTED RADIOMIC FEATURES ---")
    # Get the DataFrame with only the features selected by LASSO (with coefficients and Hazard Ratio)
    # We pass data_processor.feature_names to map the indices correctly to the feature names
    df_selected_features = lasso_cox.get_selected_features(data_processor.feature_names)
    
    # Show the first rows of the most important features
    logger.info("\nTop Selected Features:")
    logger.info(f"\n{df_selected_features.head().to_string()}")
    
    # Save the selected features to a CSV 
    output_directory = Path(settings.RESULTS_PATH) 
    output_directory.mkdir(parents=True, exist_ok=True)
    df_selected_features.to_csv(output_directory / "features_selected.csv", index=False, float_format="%.4f")

    logger.info("\n--- COX STEP 3: RISK SCORES AND SURVIVAL MONTHS PREDICTION ---")
    logger.info(f"Computing risk scores and hazards for the test set using the trained Lasso-Cox model...")
    
    risk_scores = lasso_cox.compute_risk_scores(X_input=X_test)
    hazards = lasso_cox.compute_hazards(X_input=X_test, risk_scores=risk_scores)

    # Predict the median survival months by reusing the recently generated curves
    pred_days, pred_months, survival_curves = lasso_cox.predict_survival_time(X_test=X_test)

    logger.info("\n--- COX STEP 4: MODEL EVALUATION (C-INDEX & RESIDUALS) ---")
    # Evaluate the model's performance on the test set using the C-index
    c_index_test = lasso_cox.evaluate_model(X_test, y_test)
    
    # Compute residuals and save them to a CSV file for error analysis (just for patient with Event_status=True)
    df_predictions = lasso_cox.compute_residuals_and_metrics(
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test, 
        patientID=settings.patientID, 
        risk_scores=risk_scores
    )
    
    # Saving CSV file with the associated errors for each patient 
    # (only those with Event_Status = 1, i.e., those who had the event and thus have a real survival time to compare with)
    if df_predictions is not None:
        df_predictions.to_csv(output_directory / "cox_test_set_predictions.csv", index=False, float_format="%.2f")

    logger.info("\n--- COX STEP 5: VISUALIZING EXTREME SURVIVAL CURVES ---")
    logger.info(f"Plotting and saving survival curves for patients (highest vs lowest risk) to: {settings.PLOT_SURVIVAL_CURVES}")
    
    # [UTILS] Use the plot function with survival curves and risk scores
    utils.plot_extreme_survival_curves(
        survival_functions=survival_curves, 
        risk_scores=risk_scores, 
        output_path=settings.PLOT_SURVIVAL_CURVES
    )

    logger.info("\n--- COX STEP 6: INTEGRATED BRIER SCORE (IBS) ---")
    # Compute Integrated Brier Score
    ibs= lasso_cox.evaluate_IBS(y_train, y_test, survival_curves)
    
    # Diagnostic residuals of Cox Model
    df_residuals = lasso_cox.compute_martingale_and_deviance_residuals(
        X_input=X_test,
        y_input=y_test,
        patient_ids=data_processor.patient_ids_test,
        patientID=settings.patientID, 
        risk_scores = risk_scores, 
        hazards = hazards
    )
    
    if df_residuals is not None:
        logger.info("\n--- COX STEP 7: SAVING DIAGNOSTIC RESIDUALS ---")
        residuals_path = output_directory / "cox_test_set_martingale_deviance_residuals.csv"
        df_residuals.to_csv(residuals_path, index=False, float_format="%.2f")
        logger.info(f"Diagnostic residuals of Cox model saved in: {residuals_path}")
        # >>> GENERATE THE PLOT <<<
        logger.info("\n--- COX STEP 8: PLOTTING DIAGNOSTIC RESIDUALS ---")
        utils.plot_deviance_residuals(
            df_risk_residuals=df_residuals,
            output_path=settings.PLOT_DEV_RESIDUALS
        )

    logger.info("\n--- SUMMARY: COX MODEL WORST AND BEST PREDICTIONS ---")
    # Worst and Best Cases (where the model did the biggest and the smallest error)
    if df_predictions is not None:
        worst_predictions = df_predictions.sort_values(by='Absolute_Error_Days', ascending=False)
        col = [settings.patientID, 'Actual_Days', 'Predicted_Median_Days', 'Absolute_Error_Days']
        logger.info("\nPatients with the biggest temporal prediction error:")
        logger.info(worst_predictions[col].head(5).to_string(index=False))

        best_predictions = df_predictions.sort_values(by='Absolute_Error_Days', ascending=True)
        logger.info("\nPatients with the smallest temporal prediction error:")
        logger.info(best_predictions[col].head(5).to_string(index=False))

    logger.info("\n[SUCCESS] Cox Model Elaboration completed!")

    logger.info("\n" + "=" * 100)
    logger.info(" MODEL 2: DEEP COX MODEL".center(100, " "))
    logger.info("=" * 100)

    logger.info("\n--- DEEP COX STEP 1: TRAINING DEEP COX MODEL ---")
    logger.info(f"Training Deep Cox model on the training set with {X_train.shape[0]} patients and {X_train.shape[1]} features...")
    input_dimension = X_train.shape[1]    # Number of features 

    deep_cox = DeepCoxModel(
        input_dim=input_dimension, 
        hidden_dims=args.hidden_dims, 
        dropout_rate=0.1, 
        lr=5e-4, 
        weight_decay=1e-4
    )  

    deep_cox.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)    # with 100 patients, 170 epochs and batch size 32 is a good compromise between training time and performance
    
    loss_funct_path = settings.PLOT_PATH / "deep_cox_loss_curve.png"
    utils.plot_loss_funct(deep_cox.loss_history, loss_funct_path)

    logger.info("\n--- DEEP COX STEP 2: RISK SCORES ---")
    logger.info(f"Computing risk scores and hazards for the test set using the trained Deep Cox model...")
    deep_risk_scores = deep_cox.compute_risk_scores(X_input=X_test)
    deep_hazards = deep_cox.compute_hazards(X_input=X_test, risk_scores=deep_risk_scores)

    logger.info("\n--- DEEP COX STEP 3: MODEL EVALUATION (C-INDEX) ---")
    # Evaluation of the Deep Cox (C-Index)
    deep_c_index = deep_cox.evaluate_model(X_test, y_test, risk_scores=deep_risk_scores)

    logger.info("\n--- DEEP COX STEP 4: SURVIVAL MONTHS PREDICTION ---")
    logger.info(f"Predicting survival time for the test set using the trained Deep Cox model...")
    deep_pred_days, deep_pred_months, deep_survival_curves = deep_cox.predict_survival_time(
        X_test=X_test, 
        risk_scores=deep_risk_scores, 
        hazards=deep_hazards
        )
       
    logger.info("\n--- DEEP COX STEP 5: VISUALIZING EXTREME SURVIVAL CURVES ---")
    utils.plot_extreme_survival_curves(
        survival_functions=deep_survival_curves, 
        risk_scores=deep_risk_scores, 
        output_path=settings.PLOT_DEEP_SURVIVAL_CURVES
    )
    logger.info(f"Plotting and saving Deep Cox extreme survival curves for patients (highest vs lowest risk) to: {settings.PLOT_DEEP_SURVIVAL_CURVES}")
    
    logger.info("\n--- DEEP COX STEP 6: MODEL EVALUATION (RESIDUALS) ---")
    df_deep_predictions = deep_cox.compute_residuals_and_metrics( 
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test, 
        patientID=settings.patientID, 
        risk_scores=deep_risk_scores
    )
    
    # Saving CSV file
    if df_deep_predictions is not None:
        df_deep_predictions.to_csv(output_directory / "deep_cox_test_set_predictions.csv", index=False, float_format="%.2f")

    logger.info("\n--- DEEP COX STEP 7: INTEGRATED BRIER SCORE ---")
    deep_ibs = deep_cox.evaluate_IBS(y_train, y_test, deep_survival_curves)

    # 9. Calculate diagnostic residuals (Martingala e Deviance) for each patient.
    logger.info("\n--- DEEP COX STEP 8: MARTINGALE AND DEVIANCE DIAGNOSTIC RESIDUALS ---")
    df_deep_residuals = deep_cox.compute_martingale_and_deviance_residuals(
        X_input=X_test, 
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test,
        patientID=settings.patientID, 
        risk_scores=deep_risk_scores, 
        hazards=deep_hazards
    )

    # Save the diagnostic residuals of Deep Cox to a CSV file
    if df_deep_residuals is not None:
        deep_residuals_path = output_directory / "deep_cox_martingale_deviance_residuals.csv"
        df_deep_residuals.to_csv(deep_residuals_path, index=False, float_format="%.2f")
        logger.info(f"Diagnostic residuals of Deep Cox saved in: {deep_residuals_path}")
        logger.info(f"\n--- DEEP COX STEP 9: DEEP COX PLOTTING RESIDUALS DIAGNOSTICS")
        utils.plot_deviance_residuals(
            df_risk_residuals=df_deep_residuals, 
            output_path=settings.PLOT_DEEP_DEV_RESIDUALS
        )
    
    logger.info("\n--- SUMMARY: DEEP COX MODEL WORST AND BEST PREDICTIONS ---")
    if df_deep_predictions is not None:
        deep_worst = df_deep_predictions.sort_values(by='Absolute_Error_Days', ascending=False)
        col = [settings.patientID, 'Actual_Days', 'Predicted_Median_Days', 'Absolute_Error_Days']
        logger.info(f"\nDeep Cox: Patients with the biggest temporal prediction error:")
        logger.info(deep_worst[col].head(5).to_string(index=False))

        deep_best = df_deep_predictions.sort_values(by='Absolute_Error_Days', ascending=True)
        logger.info(f"\nDeep Cox: Patients with the smallest temporal prediction error:")
        logger.info(deep_best[col].head(5).to_string(index=False))

    logger.info("\n[SUCCESS] Deep Cox Model Elaboration completed!")

    logger.info("\n" + "=" * 100)
    logger.info(" RISK CLASSIFICATION - LASSO-COX MODEL".center(100, " "))
    logger.info("=" * 100)

    classifier_cox = SurvivalRiskClassifier(trained_model=lasso_cox)
    classifier_cox.fit_threshold(X_train)
    y_pred_cox_classes = classifier_cox.predict_risk_class(risk_scores=risk_scores)
    p_value_cox = classifier_cox.evaluate_stratification(y_test=y_test, y_pred_class=y_pred_cox_classes, title_suffix="Lasso-Cox")
    
    utils.kaplan_meier_plot(
    y_test=y_test,
    pred_classes=y_pred_cox_classes,
    logrank_p_value=p_value_cox,
    title_suffix="Cox",
    output_path=settings.PLOT_PATH / "KM_popolazione_cox.png"
)

    logger.info("--- CLASSIFICATION REPORT COX---")
    matrix = classifier_cox.compute_classification_report(
    y_test=y_test, 
    y_train=y_train, 
    y_pred_class=y_pred_cox_classes
    )
    
    logger.info("--- PREDICTION REPORT COX ---")
    df_patients_class_cox = classifier_cox.generate_prediction_report(
        patient_ids=data_processor.patient_ids_test, 
        y_test=y_test, 
        predicted_time=pred_days,
        y_pred_class=y_pred_cox_classes,
        risk_scores_test=risk_scores
        )
    
    cox_class_dir = output_directory / "classes_cox.csv"
    df_patients_class_cox.to_csv(cox_class_dir, index=False, float_format="%.4f")
    logger.info(f"Classification Cox predictions saved in {cox_class_dir}")

    logger.info("\n" + "=" * 100)
    logger.info(" RISK CLASSIFICATION - DEEP COX MODEL".center(100, " "))
    logger.info("=" * 100)

    classifier_deep = SurvivalRiskClassifier(trained_model=deep_cox)
    classifier_deep.fit_threshold(X_train)
    y_pred_deep_classes = classifier_deep.predict_risk_class(risk_scores=deep_risk_scores)
    p_value_deep = classifier_deep.evaluate_stratification(y_test=y_test, y_pred_class=y_pred_deep_classes, title_suffix="DeepCox")
    
    utils.kaplan_meier_plot(
    y_test=y_test,
    pred_classes=y_pred_deep_classes,
    logrank_p_value=p_value_deep,
    title_suffix="DeepCox",
    output_path=settings.PLOT_PATH / "KM_popolazione_deepcox.png"
)

    logger.info("--- CLASSIFICATION REPORT DEEP COX ---")
    classifier_deep.compute_classification_report(
    y_test=y_test, 
    y_train=y_train, 
    y_pred_class=y_pred_deep_classes
    )
    
    logger.info("\n--- PREDICTION REPORT DEEP COX ---")
    df_patients_class_deep = classifier_deep.generate_prediction_report(
        patient_ids=data_processor.patient_ids_test, 
        y_test=y_test, 
        predicted_time=deep_pred_days,
        y_pred_class=y_pred_deep_classes,
        risk_scores_test=deep_risk_scores
        )
    
    deep_cox_class_dir = output_directory / "classes_deep_cox.csv"
    df_patients_class_deep.to_csv(deep_cox_class_dir, index=False, float_format="%.4f")
    logger.info(f"Classification Deep Cox predictions saved in {deep_cox_class_dir}")


    logger.info("\n" + "=" * 100)
    logger.info("\nPipeline executed successfully!")

if __name__ == "__main__":
    main()



