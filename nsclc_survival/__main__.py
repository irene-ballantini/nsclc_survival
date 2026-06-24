#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import pandas as pd

from nsclc_survival import ( 
    __version__, 
    RadiomicsPreprocessor, 
    FeatureExtractor, 
    RadiomicsClinicalDataProcessor, 
    LassoCoxModel,
    DeepCoxModel, 
    SurvivalRiskClassifier
    )

from nsclc_survival._download_data import download_nsclc_radiomics_data
from nsclc_survival._organize_data import organize_dicom_data

from nsclc_survival.utils import (
    save_features_to_csv, 
    plot_loss_funct,
    plot_extreme_survival_curves, 
    plot_deviance_residuals, 
    kaplan_meier_plot
)

from nsclc_survival.settings import (
    RAW_DATA_PATH, ORGANIZED_DATA_PATH, PREPROCESSED_DATA_PATH, RADIOMICS_CONFIG_PATH, 
    RAD_FEATURES_CSV_PATH, CLINICAL_FEATURES_CSV_PATH, RESULTS_PATH, PLOT_SURVIVAL_CURVES, PLOT_DEV_RESIDUALS, 
    PLOT_DEEP_DEV_RESIDUALS, PLOT_DEEP_SURVIVAL_CURVES, PLOT_PATH,
    patientID, survival_time_col, event_status_col, stage_col, gender_col, histology_col, 
    stage_mapping, gender_mapping
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s', 
    force=True
)
logger = logging.getLogger(__name__)

#logging.basicConfig(level=logging.INFO, format='%(message)s')
#logger = logging.getLogger(__name__)

def parse_args():
    """
    To change parameters from the terminal

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser(description="NSCLC Survival Analysis Pipeline")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds for the Lasso Cross-Validation")
    parser.add_argument("--epochs", type=int, default=130, help="Epochs of training for Deep Cox")
    #parser.add_argument("--skip-extraction", action="store_true", help="Salta l'estrazione dei feature radiomici")
    return parser.parse_args()

def main (): 
    args = parse_args()
    logger.info(f"Package version: {__version__}")

     # Download and organize data (if data not already available locally and organized, otherwise it will skip these steps)
    create_setup = not (ORGANIZED_DATA_PATH.exists() and any(ORGANIZED_DATA_PATH.iterdir())) 
    if create_setup:
        logger.info("[INFO] Download and organization of DICOM data in progress...")
        download_nsclc_radiomics_data()
        organize_dicom_data(raw_path=RAW_DATA_PATH, organized_path=ORGANIZED_DATA_PATH)
    
    logger.info("\n" + "="*100)
    logger.info(" 1. RUNNING RADIOMICS PREPROCESSING ".center(100, " "))
    logger.info("="*100)

    processor = RadiomicsPreprocessor(
        organized_path=ORGANIZED_DATA_PATH, 
        preprocessed_path=PREPROCESSED_DATA_PATH
    )    

    processor.process_all_patients()
    
    # print("\n" + "=" * 100)
    # print(" 2. RUNNING FEATURE EXTRACTION  ".center(100, " "))
    # print("=" * 100)

    # fe = FeatureExtractor(config_path=RADIOMICS_CONFIG_PATH)
    
    # extracted_features = fe.extract_all_features(preprocessed_path=PREPROCESSED_DATA_PATH)

    # save_features_to_csv(features_list=extracted_features, output_path=RAD_FEATURES_CSV_PATH)

    print("\n" + "=" * 100)
    print(" 3. MODELLING ".center(100, " "))
    print("=" * 100)

    print("--- LOADING AND PROCESSING DATA ---")
    data_processor = RadiomicsClinicalDataProcessor(
        radiomics_path=RAD_FEATURES_CSV_PATH, 
        clinical_path=CLINICAL_FEATURES_CSV_PATH
    )
    
    # Load and merge data
    df_merged = data_processor.load_and_merge(patientID, stage_col, gender_col, histology_col, stage_mapping, gender_mapping)
    
    # Split and standardize data
    X_train, X_test, y_train, y_test = data_processor.split_and_standardize(
        patientID = patientID,
        survival_time_col=survival_time_col,
        event_status_col=event_status_col,
        train_size=0.8,
        random_seed=42
    )

    print("\n" + "=" * 100)
    print(" MODEL 1: LASSO-COX MODEL".center(100, " "))
    print("=" * 100)

    print("\n--- COX STEP 1: TRAINING LASSO-COX MODEL WITH CROSS-VALIDATION ---")
    # Initialize the Lasso-Cox model
    lasso_cox = LassoCoxModel(
        feature_names=data_processor.feature_names,
        stage=stage_col,
        gender=gender_col,
        histology=histology_col
    )
    
    # Execute Cross-Validation to find the optimal alpha and train the final model
    # You can change the number of folds by modifying cv (e.g., cv=5)
    lasso_cox.fit_crossval(X_train, y_train, cv=5)

    print("\n--- COX STEP 2: EXTRACTING SELECTED RADIOMIC FEATURES ---")
    # Get the DataFrame with only the features selected by LASSO (with coefficients and Hazard Ratio)
    # We pass data_processor.feature_names to map the indices correctly to the feature names
    df_selected_features = lasso_cox.get_selected_features(data_processor.feature_names)
    
    # Show the first rows of the most important features
    print("\nTop Selected Features:")
    print(df_selected_features.head())
    
    # Save the selected features to a CSV 
    output_directory = Path(RESULTS_PATH)
    output_directory.mkdir(parents=True, exist_ok=True)
    df_selected_features.to_csv(output_directory / "features_selected.csv", index=False, float_format="%.4f")

    print("\n--- COX STEP 3: RISK SCORES AND SURVIVAL MONTHS PREDICTION ---")
    print(f"[INFO] Computing risk scores and hazards for the test set using the trained Lasso-Cox model...")
    
    risk_scores = lasso_cox.compute_risk_scores(X_input=X_test)
    hazards = lasso_cox.compute_hazards(X_input=X_test, risk_scores=risk_scores)

    # Predict the median survival months by reusing the recently generated curves
    pred_days, pred_months, survival_curves = lasso_cox.predict_survival_time(X_test=X_test)

    print("\n--- COX STEP 4: MODEL EVALUATION (C-INDEX & RESIDUALS) ---")
    # Evaluate the model's performance on the test set using the C-index
    c_index_test = lasso_cox.evaluate_model(X_test, y_test)
    
    # Compute residuals and save them to a CSV file for error analysis (just for patient with Event_status=True)
    df_predictions = lasso_cox.compute_residuals_and_metrics(
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test, 
        patientID=patientID, 
        risk_scores=risk_scores
    )
    
    # Saving CSV file with the associated errors for each patient 
    # (only those with Event_Status = 1, i.e., those who had the event and thus have a real survival time to compare with)
    if df_predictions is not None:
        df_predictions.to_csv(output_directory / "cox_test_set_predictions.csv", index=False, float_format="%.2f")

    print("\n--- COX STEP 5: VISUALIZING EXTREME SURVIVAL CURVES ---")
    print(f"[INFO] Plotting and saving survival curves for patients (highest vs lowest risk) to: {PLOT_SURVIVAL_CURVES}")
    
    # [UTILS] Use the plot function with survival curves and risk scores
    plot_extreme_survival_curves(
        survival_functions=survival_curves, 
        risk_scores=risk_scores, 
        output_path=PLOT_SURVIVAL_CURVES
    )

    print("\n--- COX STEP 6: INTEGRATED BRIER SCORE (IBS) ---")
    # Compute Integrated Brier Score
    ibs= lasso_cox.evaluate_IBS(y_train, y_test, survival_curves)
    
    # Diagnostic residuals of Cox Model
    df_residuals = lasso_cox.compute_martingale_and_deviance_residuals(
        X_input=X_test,
        y_input=y_test,
        patient_ids=data_processor.patient_ids_test,
        patientID=patientID, 
        risk_scores = risk_scores, 
        hazards = hazards
    )
    
    if df_residuals is not None:
        print("\n--- COX STEP 7: SAVING DIAGNOSTIC RESIDUALS ---")
        residuals_path = output_directory / "cox_test_set_martingale_deviance_residuals.csv"
        df_residuals.to_csv(residuals_path, index=False, float_format="%.2f")
        print(f"[INFO] Diagnostic residuals of Cox model saved in: {residuals_path}")
        # >>> GENERATE THE PLOT <<<
        print("\n--- COX STEP 8: PLOTTING DIAGNOSTIC RESIDUALS ---")
        plot_deviance_residuals(
            df_risk_residuals=df_residuals,
            output_path=PLOT_DEV_RESIDUALS
        )

    print("\n--- SUMMARY: COX MODEL WORST AND BEST PREDICTIONS ---")
    # Worst and Best Cases (where the model did the biggest and the smallest error)
    if df_predictions is not None:
        worst_predictions = df_predictions.sort_values(by='Absolute_Error_Days', ascending=False)
        col = [patientID, 'Actual_Days', 'Predicted_Median_Days', 'Absolute_Error_Days']
        print("\nPatients with the biggest temporal prediction error:")
        print(worst_predictions[col].head(5).to_string(index=False))

        best_predictions = df_predictions.sort_values(by='Absolute_Error_Days', ascending=True)
        print("\nPatients with the smallest temporal prediction error:")
        print(best_predictions[col].head(5).to_string(index=False))

    print("\n[SUCCESS] Cox Model Elaboration completed!")
    
    print("\n" + "=" * 100)
    print(" MODEL 2: DEEP COX MODEL".center(100, " "))
    print("=" * 100)

    print("\n--- DEEP COX STEP 1: TRAINING DEEP COX MODEL ---")
    print(f"[INFO] Training Deep Cox model on the training set with {X_train.shape[0]} patients and {X_train.shape[1]} features...")
    input_dimension = X_train.shape[1]    # Number of features 

    deep_cox = DeepCoxModel(
        input_dim=input_dimension, 
        hidden_dims=[4], 
        dropout_rate=0.1, 
        lr=5e-4, 
        weight_decay=1e-4
    )  

    deep_cox.fit(X_train, y_train, epochs=170, batch_size=32)
    
    loss_funct_path = PLOT_PATH / "deep_cox_loss_curve.png"
    plot_loss_funct(deep_cox.loss_history, loss_funct_path)

    print("\n--- DEEP COX STEP 2: RISK SCORES ---")
    print(f"[INFO] Computing risk scores and hazards for the test set using the trained Deep Cox model...")
    deep_risk_scores = deep_cox.compute_risk_scores(X_input=X_test)
    deep_hazards = deep_cox.compute_hazards(X_input=X_test, risk_scores=deep_risk_scores)
    
    print("\n--- DEEP COX STEP 3: MODEL EVALUATION (C-INDEX) ---")
    # Evaluation of the Deep Cox (C-Index)
    deep_c_index = deep_cox.evaluate_model(X_test, y_test, risk_scores=deep_risk_scores)
    
    print("\n--- DEEP COX STEP 4: SURVIVAL MONTHS PREDICTION ---")
    print(f"[INFO] Predicting survival time for the test set using the trained Deep Cox model...")
    deep_pred_days, deep_pred_months, deep_survival_curves = deep_cox.predict_survival_time(
        X_test=X_test, 
        risk_scores=deep_risk_scores, 
        hazards=deep_hazards
        )
       
    print("\n--- DEEP COX STEP 5: VISUALIZING EXTREME SURVIVAL CURVES ---")
    plot_extreme_survival_curves(
        survival_functions=deep_survival_curves, 
        risk_scores=deep_risk_scores, 
        output_path=PLOT_DEEP_SURVIVAL_CURVES
    )
    print(f"[INFO] Plotting and saving Deep Cox extreme survival curves for patients (highest vs lowest risk) to: {PLOT_DEEP_SURVIVAL_CURVES}")
    
    print("\n--- DEEP COX STEP 6: MODEL EVALUATION (RESIDUALS) ---")
    df_deep_predictions = deep_cox.compute_residuals_and_metrics( 
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test, 
        patientID=patientID, 
        risk_scores=deep_risk_scores
    )
    
    # Saving CSV file
    if df_deep_predictions is not None:
        df_deep_predictions.to_csv(output_directory / "deep_cox_test_set_predictions.csv", index=False, float_format="%.2f")

    print("\n--- DEEP COX STEP 7: INTEGRATED BRIER SCORE ---")
    deep_ibs = deep_cox.evaluate_IBS(y_train, y_test, deep_survival_curves)

    # 9. Calculate diagnostic residuals (Martingala e Deviance) for each patient.
    print("\n--- DEEP COX STEP 8: MARTINGALE AND DEVIANCE DIAGNOSTIC RESIDUALS ---")
    df_deep_residuals = deep_cox.compute_martingale_and_deviance_residuals(
        X_input=X_test, 
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test,
        patientID=patientID, 
        risk_scores=deep_risk_scores, 
        hazards=deep_hazards
    )

    # Save the diagnostic residuals of Deep Cox to a CSV file
    if df_deep_residuals is not None:
        deep_residuals_path = output_directory / "deep_cox_martingale_deviance_residuals.csv"
        df_deep_residuals.to_csv(deep_residuals_path, index=False, float_format="%.2f")
        print(f"[INFO] Diagnostic residuals of Deep Cox saved in: {deep_residuals_path}")
        print(f"\n--- DEEP COX STEP 9: DEEP COX PLOTTING RESIDUALS DIAGNOSTICS")
        plot_deviance_residuals(
            df_risk_residuals=df_deep_residuals, 
            output_path=PLOT_DEEP_DEV_RESIDUALS
        )
    
    print("\n--- SUMMARY: DEEP COX MODEL WORST AND BEST PREDICTIONS ---")
    if df_deep_predictions is not None:
        deep_worst = df_deep_predictions.sort_values(by='Absolute_Error_Days', ascending=False)
        col = [patientID, 'Actual_Days', 'Predicted_Median_Days', 'Absolute_Error_Days']
        print("\nDeep Cox: Patients with the biggest temporal prediction error:")
        print(deep_worst[col].head(5).to_string(index=False))

        deep_best = df_deep_predictions.sort_values(by='Absolute_Error_Days', ascending=True)
        print("\nDeep Cox: Patients with the smallest temporal prediction error:")
        print(deep_best[col].head(5).to_string(index=False))

    print("\n[SUCCESS] Deep Cox Model Elaboration completed!")

    print("\n" + "=" * 100)
    print(" RISK CLASSIFICATION - LASSO-COX MODEL".center(100, " "))
    print("=" * 100)

    classifier_cox = SurvivalRiskClassifier(trained_model=lasso_cox)
    classifier_cox.fit_threshold(X_train)
    y_pred_cox_classes = classifier_cox.predict_risk_class(risk_scores=risk_scores)
    p_value_cox = classifier_cox.evaluate_stratification(y_test=y_test, y_pred_class=y_pred_cox_classes, title_suffix="Lasso-Cox")
    
    kaplan_meier_plot(
    y_test=y_test,
    pred_classes=y_pred_cox_classes,
    logrank_p_value=p_value_cox,
    title_suffix="Cox",
    output_path=PLOT_PATH / "KM_popolazione_cox.png"
)

    print("--- CLASSIFICATION REPORT COX---")
    matrix = classifier_cox.compute_classification_report(
    y_test=y_test, 
    y_train=y_train, 
    y_pred_class=y_pred_cox_classes
    )
    
    print("--- PREDICTION REPORT COX ---")
    df_patients_class_cox = classifier_cox.generate_prediction_report(
        patient_ids=data_processor.patient_ids_test, 
        y_test=y_test, 
        predicted_time=pred_days,
        y_pred_class=y_pred_cox_classes,
        risk_scores_test=risk_scores
        )
    
    cox_class_dir = output_directory / "classes_cox.csv"
    df_patients_class_cox.to_csv(cox_class_dir, index=False, float_format="%.4f")
    print(f"[INFO] Classification Cox predictions saved in {cox_class_dir}")

    print("\n" + "=" * 100)
    print(" RISK CLASSIFICATION - DEEP COX MODEL".center(100, " "))
    print("=" * 100)

    classifier_deep = SurvivalRiskClassifier(trained_model=deep_cox)
    classifier_deep.fit_threshold(X_train)
    y_pred_deep_classes = classifier_deep.predict_risk_class(risk_scores=deep_risk_scores)
    p_value_deep = classifier_deep.evaluate_stratification(y_test=y_test, y_pred_class=y_pred_deep_classes, title_suffix="DeepCox")
    
    kaplan_meier_plot(
    y_test=y_test,
    pred_classes=y_pred_deep_classes,
    logrank_p_value=p_value_deep,
    title_suffix="DeepCox",
    output_path=PLOT_PATH / "KM_popolazione_deepcox.png"
)

    print("--- CLASSIFICATION REPORT DEEP COX ---")
    classifier_deep.compute_classification_report(
    y_test=y_test, 
    y_train=y_train, 
    y_pred_class=y_pred_deep_classes
    )
    
    print("\n--- PREDICTION REPORT DEEP COX ---")
    df_patients_class_deep = classifier_deep.generate_prediction_report(
        patient_ids=data_processor.patient_ids_test, 
        y_test=y_test, 
        predicted_time=deep_pred_days,
        y_pred_class=y_pred_deep_classes,
        risk_scores_test=deep_risk_scores
        )
    
    deep_cox_class_dir = output_directory / "classes_deep_cox.csv"
    df_patients_class_deep.to_csv(deep_cox_class_dir, index=False, float_format="%.4f")
    print(f"[INFO] Classification Deep Cox predictions saved in {deep_cox_class_dir}")


    print("\n" + "=" * 100)
    print("\nPipeline executed successfully!")

if __name__ == "__main__":
    main()



