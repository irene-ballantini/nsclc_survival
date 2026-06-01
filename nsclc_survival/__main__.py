#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

from nsclc_survival import ( 
    __version__, 
    RadiomicsPreprocessor, 
    FeatureExtractor, 
    RadiomicsClinicalDataProcessor, 
    LassoCoxModel
    )

from nsclc_survival._download_data import download_nsclc_radiomics_data
from nsclc_survival._organize_data import organize_dicom_data

from nsclc_survival.utils import (
    save_features_to_csv, 
    plot_extreme_survival_curves, 
    plot_deviance_residuals
)

from nsclc_survival.settings import (
    RAW_DATA_PATH, ORGANIZED_DATA_PATH, PREPROCESSED_DATA_PATH, RADIOMICS_CONFIG_PATH, 
    RAD_FEATURES_CSV_PATH, CLINICAL_FEATURES_CSV_PATH, RESULTS_PATH, PLOT_SURVIVAL_CURVES, PLOT_DEV_RESIDUALS, 
    patientID, survival_time_col, event_status_col, stage_col, gender_col, histology_col, 
    stage_mapping, gender_mapping
)

def main (): 
    print(__version__) 

     # Download and organize data (if data not already available locally and organized, otherwise it will skip these steps)
    create_setup = not (ORGANIZED_DATA_PATH.exists() and any(ORGANIZED_DATA_PATH.iterdir())) 
    if create_setup:
        download_nsclc_radiomics_data()
        organize_dicom_data(raw_path=RAW_DATA_PATH, organized_path=ORGANIZED_DATA_PATH)
    
    print("\n" + "#"*50)
    print(" 1. RUNNING RADIOMICS PREPROCESSING ".center(50, "#"))
    print("#"*50)

    processor = RadiomicsPreprocessor(
        organized_path=ORGANIZED_DATA_PATH, 
        preprocessed_path=PREPROCESSED_DATA_PATH
    )    

    processor.process_all_patients()
    
    # print("\n" + "#" * 50)
    # print(" 2. RUNNING FEATURE EXTRACTION  ".center(50, "#"))
    # print("#" * 50)

    # fe = FeatureExtractor(config_path=RADIOMICS_CONFIG_PATH)
    
    # extracted_features = fe.extract_all_features(preprocessed_path=PREPROCESSED_DATA_PATH)

    # save_features_to_csv(features_list=extracted_features, output_path=RAD_FEATURES_CSV_PATH)

    print("\n" + "#" * 50)
    print(" 3. COX MODEL".center(50, "#"))
    print("#" * 50)
    
    print("--- STEP 1: LOADING AND PROCESSING DATA ---")
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

    print("\n--- STEP 2: TRAINING LASSO-COX MODEL WITH CROSS-VALIDATION ---")
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

    print("\n--- STEP 3: MODEL EVALUATION ---")
    # Evaluate the model's performance on the test set using the C-index
    c_index_test = lasso_cox.evaluate_model(X_test, y_test)

    print("\n--- STEP 4: EXTRACTING SELECTED RADIOMIC FEATURES ---")
    # Get the DataFrame with only the features selected by LASSO (with coefficients and Hazard Ratio)
    # We pass data_processor.feature_names to map the indices correctly to the feature names
    df_selected_features = lasso_cox.get_selected_features(data_processor.feature_names)
    
    # Show the first rows of the most important features
    print("\nTop Selected Features:")
    print(df_selected_features.head())

    output_directory = Path(RESULTS_PATH)
    output_directory.mkdir(parents=True, exist_ok=True)
    df_selected_features.to_csv(output_directory / "features_selected.csv", index=False, float_format="%.4f")

    print("\n--- STEP 5: SURVIVAL MONTHS PREDICTION AND RISK SCORE---")
    
    # Predict the median survival months by reusing the recently generated curves
    predicted_days, predicted_months, survival_curves = lasso_cox.predict_survival_time(
        X_test=X_test
    )

    df_risk_scores = lasso_cox.compute_risk_scores(
        X_input=X_test, 
        y_input=y_test, 
        patient_ids=data_processor.patient_ids_test, 
        patientID=patientID, 
        predicted_medians_d=predicted_days, 
        predicted_medians_m=predicted_months

    )

    print(f"\nPredicted Median Survival Months for the Test Set Patients:")
    print(df_risk_scores[[patientID, 'Predicted_Median_Months']].head())
    
    # Saving the results of the predictions in a CSV file
    df_risk_scores.to_csv(output_directory / "test_set_predictions.csv", index=False, float_format="%.2f")
    
    df_residuals = lasso_cox.compute_residuals_and_metrics(df_risk_scores, patientID=patientID)
    
    # Saving CSV file with the associated errors 
    if df_residuals is not None:
        df_residuals.to_csv(output_directory / "test_set_predictions_residuals.csv", index=False, float_format="%.2f")

    print("\n--- STEP 6: VISUALIZING EXTREME SURVIVAL CURVES ---")
    print("[INFO] Plotting survival curves for patients at highest vs lowest risk...")
    
    # [UTILS] Use the plot function with survival curves and risk scores
    # Note: extract the risk score array from the dataframe
    plot_extreme_survival_curves(
        survival_functions=survival_curves, 
        risk_scores=df_risk_scores['Risk_Score'].values, 
        output_path=PLOT_SURVIVAL_CURVES
    )

    # Compute Integrated Brier Score
    ibs= lasso_cox.evaluate_IBS(y_train, y_test, survival_curves)

    # Worst Cases (where the model did the biggest value)
    if df_residuals is not None:
        worst_predictions = df_residuals.sort_values(by='Absolute_Error_Days', ascending=False)
        col = [patientID, 'Actual_Days', 'Predicted_Median_Days', 'Absolute_Error_Days']
        print("\nPatients with the biggest temporal prediction error:")
        print(worst_predictions[col].head(5).to_string(index=False))

        best_predictions = df_residuals.sort_values(by='Absolute_Error_Days', ascending=True)
        print("\nPatients with the smallest temporal prediction error:")
        print(best_predictions[col].head(5).to_string(index=False))

    # Individual Error on Risk Score
    df_risk_residuals = lasso_cox.compute_martingale_and_deviance_residuals(
        X_input=X_test,
        y_input=y_test,
        patient_ids=data_processor.patient_ids_test,
        patientID=patientID, 
        df_risk = df_risk_scores
    )
    
    if df_risk_residuals is not None:
        df_risk_residuals.to_csv(output_directory / "test_set_risk_residuals.csv", index=False, float_format="%.2f")
        # >>> GENERIAMO IL GRAFICO <<<
        print("\n--- STEP 7: PLOTTING RESIDUALS DIAGNOSTICS ---")
        plot_deviance_residuals(
            df_risk_residuals=df_risk_residuals,
            output_path=PLOT_DEV_RESIDUALS
        )


    print("\nPipeline executed successfully!")

if __name__ == "__main__":
    main()



