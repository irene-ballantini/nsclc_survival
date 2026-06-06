#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, RepeatedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. DATA LOADING AND PROCESSING
# ==========================================
class RadiomicsClinicalDataProcessor:
    """
    Class to load data, merge radiomics features with clinical data, 
    split data into train and test sets and standardize data. 

    This class handles the pipeline from raw CSV data to processed matrices 
    ready for survival modeling, ensuring proper feature scaling and 
    preventing data leakage.

    Attributes:
        radiomics_path (str): File path to the radiomics features CSV file.
        clinical_path (str): File path to the clinical data CSV file.
        df_total (pandas.DataFrame or None): Combined dataframe containing both 
            radiomics and clinical variables after merging.
        X_train (numpy.ndarray or None): Standardized training feature matrix.
        X_test (numpy.ndarray or None): Standardized testing feature matrix.
        y_train (numpy.ndarray or None): Structured array of training survival 
            targets (Event_Status, Survival_Time).
        y_test (numpy.ndarray or None): Structured array of testing survival 
            targets (Event_Status, Survival_Time).
        scaler (StandardScaler): Scikit-learn scaler object for Z-score 
            standardization.
        feature_names (pandas.Index or None): Names of the features included 
            in the X matrices.
        patient_ids_train(numpy.ndarray or None): 1D array of strings containing the 
            PatientIDs for the training set samples, maintaining exact row alignment.
        patient_ids_test(numpy.ndarray or None): 1D array of strings containing the 
            PatientIDs for the testing set samples, used to map predictions back to 
            specific patients during final evaluation.
        
    """
    def __init__(self, radiomics_path, clinical_path):
        self.radiomics_path = radiomics_path
        self.clinical_path = clinical_path
        self.df_total = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.scaler = StandardScaler()
        self.feature_names = None

        self.patient_ids_train = None
        self.patient_ids_test = None

    def load_and_merge(self, patientID, stage, gender, histology, stage_mapping, gender_mapping):
        """
        Load the radiomics and clinical CSV files and merges them on PatientID.

        Args:
            patientID (str): Name of the column containing patient identifiers,
            stage (str): Name of the column containing cancer stage information.
            gender (str): Name of the column containing gender information.
            histology (str): Name of the column containing histology information.
            stage_mapping (dict): Mapping dictionary for cancer stage values.
            gender_mapping (dict): Mapping dictionary for gender values.

        Returns:
            pandas.DataFrame: The merged dataset containing both radiomic 
                features and clinical records.
        """
        # Load radiomics features
        df_radiomics = pd.read_csv(self.radiomics_path)
        df_clinical = pd.read_csv(self.clinical_path)
        self.df_total = pd.merge(df_radiomics, df_clinical, on=patientID)
        print(f"Data merging completed. Total rows: {self.df_total.shape[0]}")
        print(f"Unique patients: {self.df_total[patientID].nunique()}")    # there could be duplicates if there are multiple records per patient
        
        # 1. Stage mapping
        if stage in self.df_total.columns:
            self.df_total[stage] = self.df_total[stage].map(stage_mapping).fillna(3)  # 3 as modal fallback
            
        # 2. Gender mapping
        if gender in self.df_total.columns:
            self.df_total[gender] = self.df_total[gender].map(gender_mapping).fillna(1)
            
        # 3. Histology mapping (One-Hot Encoding since there's no hierarchical order)
        if histology in self.df_total.columns:
            self.df_total[histology] = self.df_total[histology].fillna('Unknown')
            self.df_total = pd.get_dummies(self.df_total, columns=[histology], drop_first=True, dtype=int)

        return self.df_total
    
    def split_and_standardize(self, patientID, survival_time_col, event_status_col, train_size=0.8, random_seed=42):
        """
        Separate predictors in X from targets in y, perform a stratified train/test split, and applies the Z-score standardization.
        
        Args:
            patientID (str): Name of the column containing patient identifiers.
            survival_time_col (str): Column name representing the time-to-event (Death/Event) or 
                last follow-up time(censoring).
            event_status_col (str): Column name representing the binary event occurrence 
                (e.g., 1 for death/recurrence, 0 for censored (patient still alive at last follow-up)).
                It is then converted to a boolean format (True for event, False for censored).
            train_size (float, optional): Proportion of the dataset to include in the train split. Defaults to 0.8.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If 'load_and_merge()' has not been executed prior to calling 
                this method.

        Returns:
            tuple: A tuple containing four elements:
                - X_train (numpy.ndarray): Scaled training feature matrix.
                - X_test (numpy.ndarray): Scaled testing feature matrix.
                - y_train (numpy.ndarray): Structured array of training targets.
                - y_test (numpy.ndarray): Structured array of testing targets.
        """

        if self.df_total is None:
            raise ValueError("Call load_and_merge() before splitting and standardizing.")
        
        # Separation of X: 
        # we exclude the ID (since it's just an identifier and doesn't have predictive value) 
        # and both survival clinical data (to prevent data leakage).
        # Survival time and event status will be predicted by the model.

        X = self.df_total.select_dtypes(include=[np.number]).drop(
            columns=[survival_time_col, event_status_col, patientID], errors='ignore'
        )

        # Creation of structured array for sksurv (y)
        # Convert the event to a boolean (True if death, False if censored = the patient is still alive at the last follow-up)
        y = np.array(
            list(zip(self.df_total[event_status_col].astype(bool), self.df_total[survival_time_col])),
            dtype=[('Event_Status', '?'), ('Survival_Time', '<f8')]
        )
        
        #Check for duplicates in PatientID to ensure that there's no data leakage between train and test sets. 
        has_duplicates = self.df_total[patientID].nunique() < self.df_total.shape[0]
        
        if not has_duplicates:
            # CASE A: No duplicates -> Priority to stratification to ensure balanced representation of outcomes in train and test sets
            print("INFO: No duplicates found. Applying stratified split")

            try:
                # Stratification based on both Event Status and Survival Time 
                # 4 classes created (0_False, 0_True, 1_False, 1_True)
                # Possible when each group has more than 1 sample)
                time_median = self.df_total[survival_time_col].median()
                stratify_col = (
                    self.df_total[event_status_col].astype(str) + "_" + 
                    (self.df_total[survival_time_col] > time_median).astype(str)
                )
            
                X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
                    X, y, train_size=train_size, random_state=random_seed, stratify=stratify_col
                )

            except ValueError:
                print("INFO: Dataset too small. Stratification based on Event Status only.")
                X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
                    X, y, train_size=train_size, random_state=random_seed, stratify=self.df_total[event_status_col]
                )  

            self.patient_ids_train = self.df_total.loc[X_train_raw.index, patientID].values
            self.patient_ids_test = self.df_total.loc[X_test_raw.index, patientID].values

        else:
            # CASE B: duplicates present -> Maximum priority to avoid DATA LEAKAGE (Groups)
            print("WARNING: Duplicates found for patients. Priority given to Group Management (No Data Leakage).")
            gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_seed)
            train_idx, test_idx = next(gss.split(X, y, groups=self.df_total[patientID]))
            
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            self.y_train, self.y_test = y[train_idx], y[test_idx]
            
            self.patient_ids_train = self.df_total.iloc[train_idx][patientID].values
            self.patient_ids_test = self.df_total.iloc[test_idx][patientID].values
        
        # imputer for completing missing values using a descriptive statistic (e.g. median)
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_test_imputed = imputer.transform(X_test_raw)

        # Standardization (Z-score) of features: fit the scaler on the training data and transform both train and test sets
        self.X_train = self.scaler.fit_transform(X_train_imputed)
        self.X_test = self.scaler.transform(X_test_imputed)
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"Split completed. Train size: {self.X_train.shape[0]}, Test size: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
# ==========================================
# 2. MODEL CLASS: LASSO-COX
# ==========================================
class LassoCoxModel:
    """
    Handles training, hyperparameter alpha optimization, and evaluation of a LASSO-regularized regression Cox model.

    This class selects the radiomics features that are most relevant for predicting survival outcomes, 
    while also providing a measure of model performance through the C-index and other metrics.  

    Attributes:
        model (CoxnetSurvivalAnalysis or None): The final estimator optimized
            via Cross-Validation. Initially set to None.
        best_alpha (float or None): The optimal alpha penalty parameter found 
            during optimization. Initially set to None.
        feature_names (list of str): List of all feature names corresponding to the columns in the input X matrices.
        stage (str): Name of the column containing cancer stage information.
        gender (str): Name of the column containing gender information.
        histology (str): Name of the column containing histology information.
        predicted_medians_d (numpy.ndarray or None): 1D array of predicted median survival times in days. 
            Populated after running predict_survival_time().
        predicted_medians_m (numpy.ndarray or None): 1D array of predicted median survival times in months. 
            Populated after running predict_survival_time().
    """
    def __init__(self, feature_names, stage, gender, histology):
        self.model = None
        self.best_alpha = None
        self.feature_names = feature_names
        self.stage = stage
        self.gender = gender
        self.histology = histology

        self.predicted_medians_d = None
        self.predicted_medians_m = None

    def fit_crossval(self, X_train, y_train, cv=5, n_repeats=3, alpha_param_grid=None):     
        """
        Optimizes the alpha hyperparameter using Repeated K-Fold Cross-Validation 
        and GridSearchCV, then trains the final LASSO-Cox model on the entire training set.

        The optimization maximizes Harrell's Concordance Index (C-index). Clinical variables 
        are protected from the LASSO penalty using custom penalty factors, ensuring they are 
        never shrunk to zero.

        Args:
            X_train (numpy.ndarray): Standardized training feature matrix of shape 
                (n_samples, n_features).
            y_train (numpy.ndarray): NumPy structured array containing the survival target 
                with two fields: 'Event_Status' (bool) and 'Survival_Time' (float).
            cv (int, optional): Number of folds for the K-Fold Cross-Validation. 
                Defaults to 5.
            n_repeats (int, optional): Number of times Cross-Validation is repeated 
                with different random splits. Defaults to 3.
            alpha_param_grid (dict, optional): Dictionary containing the grid of alpha values 
                to test. Format: {"alphas": [[a1], [a2], ...]}. If None, a logarithmic grid 
                between 10^-1.0 and 10^-0.1 with 20 values is automatically generated. 
                Defaults to None.
        """
        print("Starting optimized research through GridSearchCV...")
        
        # Use Lasso
        l1_ratio_chosen = 1.0
        
        # Define a logarithmic grid for alphas.
        # Avoid too big values that brings everything to 0,
        # and avoid very small values (e.g. < 1e-4) that cause the ArithmeticError because of the weights too large.
        if alpha_param_grid is None:
            alpha_param_grid= {
                "alphas": [[a] for a in np.logspace(-1.0, -0.1, num=20)]
            }
        
        # Create an array of weights for the penalty as long as the column in X 
        # 1 = normal penalty (for radiomics features)
        # 0 = no penalty(clinical variables never reset to zero by LASSO)
        penalty_factors = np.ones(X_train.shape[1])

        # Find the clinical column indexes 
        clinical = [self.stage, self.gender] + [col for col in self.feature_names if self.histology in col]

        for col in clinical:
            if col in self.feature_names:
                idx = self.feature_names.index(col)
                penalty_factors[idx] = 0.0  # in this way LASSO doesn't penalize them

        base_estimator = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio_chosen,
            max_iter=100000,
            tol=1e-3, 
            penalty_factor=penalty_factors
        )

        cv_strategy = RepeatedKFold(n_splits=cv, n_repeats=n_repeats, random_state=42)
        
        # Use GridSearchCV 
        gcv = GridSearchCV(
            estimator=base_estimator,
            param_grid=alpha_param_grid,
            cv=cv_strategy,
            error_score=0.5 # If a fold fails, it assigns the pure chance C-index (0.5)
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message=".*all coefficients are zero.*")
            gcv.fit(X_train, y_train)
        
        # Extract the optimal alpha and internal validation
        self.best_alpha = gcv.best_params_["alphas"][0]
        mean_best_score = gcv.best_score_
        best_index = gcv.best_index_
        cv_results = gcv.cv_results_
        std_best_score = cv_results['std_test_score'][best_index]    # local test fold result of CV
        
        print(f"Optimization completed. Best Alpha: {self.best_alpha:.6f}" )
        print(f"(Mean Validation C-index: {mean_best_score:.4f} \u00B1 {std_best_score:.4f})")
        
        # Train the final model
        self.model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio_chosen,
            alphas=[self.best_alpha],
            max_iter=150000,
            tol=1e-4,
            penalty_factor=penalty_factors,
            fit_baseline_model=True
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Calculate the C-index on the test set.

        Args:
            X_test (array-like or pandas.DataFrame): Test feature matrix 
                of shape (n_samples, n_features).
            y_test (NumPy structured array): Survival target for testing 
                (event status, time) used for external validation.

        Raises:
            ValueError: If the model has not been trained with fit_crossval().

        Returns:
            float: The C-index (Harrell's concordance coefficient) on the test set.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")
        c_index = self.model.score(X_test, y_test)
        print(f"LASSO-Cox C-index on Test Set: {c_index:.4f}")
        return c_index

    def get_selected_features(self, feature_names):
        """
        Extract radiomic features selected by the LASSO penalty, along with 
        their coefficients and Hazard Ratios (HR), ordered by absolute impact:

        - HR < 1 indicates a protective effect (associated with better survival),
        - HR > 1 indicates a risk factor (associated with worse survival).

        Args:
            feature_names (list of str): Full list containing the names of all original features.

        Raises:
            ValueError: If the model has not been trained with fit_crossval().

        Returns:
            pandas.DataFrame: A DataFrame containing 'Feature', 'Coefficient', 
                and 'Hazard_Ratio' for all features whose coefficients were 
                not shrunk to zero, sorted by absolute coefficient value.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")
        
        # Use .squeeze() to be sure to get a 1D array
        coefs = self.model.coef_.squeeze()

        # Create a DataFrame with Feature, Coefficient and Hazard Ratio for the specific feature
        df_features = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Hazard_Ratio': np.exp(coefs)
        })

        # Filter features with non-zero coefficients
        df_selected = df_features[df_features['Coefficient'] != 0].copy()
        
        # Order by absolute value of coefficient (most influential features at the top)
        df_selected['Abs_Coefficient'] = df_selected['Coefficient'].abs()
        # ascending=False -> descending order
        df_selected = df_selected.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])

        selected_names = df_selected['Feature'].tolist()
        print(f"Feature selected ({len(selected_names)} out of {len(feature_names)}): {selected_names}")
        
        return df_selected
    
    def compute_risk_scores(self, X_input):
        """
        Compute the risk scores (Rad-Scores) for the input data using the trained model.
        
        Args:
            X_input (array-like or pandas.DataFrame): Feature matrix of shape 
                (n_samples, n_features) for which to compute risk scores.

        Raises:
            ValueError: If the model has not been trained with fit_crossval().

        Returns:
            numpy.ndarray: 1D array of computed risk scores.
        """     

        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")

        # Compute linear index (Rad-Score)
        # Use .squeeze() to be sure to get a 1D array
        risk_scores = self.model.predict(X_input).squeeze()  
        
        return risk_scores
    
    def compute_hazards(self, X_input, risk_scores=None):
        """
        Compute the hazard rates for the input data using the trained model.

        Args:
            X_input (array-like or pandas.DataFrame): Feature matrix for which to compute hazard rates.
            risk_scores (numpy.ndarray, optional): Pre-computed risk scores. If not provided, they will be computed internally.     

        Returns:       
            numpy.ndarray: 1D array of computed hazard rates.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")
        
        if risk_scores is None:
            risk_scores = self.compute_risk_scores(X_input)
        
        # In Cox models, the hazard is proportional to exp(risk_score)
        hazards = np.exp(risk_scores)
        
        return hazards
    
    def predict_survival_time(self, X_test):
        """
        Predict the median survival time (expressed in days and months) 
        for each patient using the optimized Cox model and returns predictions 
        alongside the survival curves.

        The median survival time is defined as the time point at which the 
        survival probability drops to or below 50%. If a patient's survival curve 
        never reaches 50% within the maximum observation period, the maximum 
        available follow-up time is returned as a conservative estimate.

        Args:
            X_test (array-like or pandas.DataFrame): Feature matrix for the test samples.

        Raises:
            ValueError: If the model has not been trained with fit_crossval().

        Returns:
            tuple: (predicted_medians_d, predicted_medians_m, survival_functions) where:
                - predicted_medians_d (numpy.ndarray): 1D array of predicted median survival times in days.
                - predicted_medians_m (numpy.ndarray): 1D array of predicted median survival times in months.
                - survival_functions (list of sksurv.functions.StepFunction): Computed survival curves.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")
        
        survival_functions = self.model.predict_survival_function(X_test)
        
        predicted_medians_d = []
        average_days_per_month = 30.437
        
        for fn in survival_functions:
            # fn.x = time points (e.g., days or months), fn.y = survival probabilities
            under_50 = np.where(fn.y <= 0.5)[0]
            
            if len(under_50) > 0:
                idx = under_50[0]
                # Optional refinement: if it's strictly less than 0.5, we can approximate 
                # better by checking the step right before, or just take the first cross-point.
                median_time_d = fn.x[idx]
            else:
                # If the probability never drops below 50%, the median is mathematically undefined.
                # We fallback to the maximum observation time in the study for that curve.
                median_time_d = fn.x[-1]
            predicted_medians_d.append(median_time_d)
        
        self.predicted_medians_d = np.array(predicted_medians_d)
        self.predicted_medians_m = self.predicted_medians_d/average_days_per_month   
        
        return self.predicted_medians_d, self.predicted_medians_m, survival_functions             
    
    def compute_residuals_and_metrics(self, y_input, patient_ids, patientID, risk_scores=None, X_input=None):
        """
        Filter out censored patients keeping only those with an observed event, 
        print linear regression metrics (MAE/RMSE), and return a residuals DataFrame.

        Args:
            y_input (numpy.ndarray): The input data for the model.
            patient_ids (list): A list of patient IDs.
            patientID (str): Name of the column containing the patient identifiers.
            risk_scores (numpy.ndarray, optional): Pre-computed risk scores. If not provided, they will be computed internally.
            X_input (numpy.ndarray or pandas.DataFrame, optional): Feature matrix for computing risk scores if they are not provided.

        Raises:
            ValueError: If the model has not been trained with fit_crossval(), 
                or if predictions are missing, 
                or if risk scores cannot be computed due to missing data.

        Returns:
            pandas.DataFrame or None: A filtered DataFrame containing residuals for patients with an observed event.
                Returns None if no events are observed in the input data.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")
        
        if not hasattr(self, 'predicted_medians_d') or self.predicted_medians_d is None:
            raise ValueError("Predictions missing. Run predict_survival_time() first to populate median times.")

        # Boolean mask to filter only patients with observed events
        event_observed = y_input['Event_Status'] == True
        
        if not np.any(event_observed):
            print("[!] Time Error: No observed events found in the Test Set. Cannot compute MAE/RMSE.")
            return None

        # Extract the filtered data 

        actual_days_all = y_input['Survival_Time']
        actual_months_all = actual_days_all / 30.437
        
        actual_days = actual_days_all[event_observed]
        actual_months = actual_months_all[event_observed]

        pred_days = self.predicted_medians_d[event_observed]
        pred_months = self.predicted_medians_m[event_observed]

        patient_ids_arr = np.asarray(patient_ids)[event_observed]

        if risk_scores is not None:
            risk_scores_filtered = np.asarray(risk_scores)[event_observed]
        else:
            if X_input is None:
                raise ValueError("Missing data: You must provide either 'risk_scores' or 'X_input' to extract Rad-Scores.")
            risk_scores = self.compute_risk_scores(X_input)
            risk_scores_filtered = np.asarray(risk_scores)[event_observed]

        # Linear error metrics computation
        mae_days = mean_absolute_error(actual_days, pred_days)
        rmse_days = root_mean_squared_error(actual_days, pred_days)
        mae_months = mean_absolute_error(actual_months, pred_months)
        rmse_months = root_mean_squared_error(actual_months, pred_months)
    
        print(f"\nSurvival Time Error Analysis (N. patients with event = {len(pred_days)}):")
        print(f"    - Mean Absolute Error (MAE):   {mae_days:.2f} days | {mae_months:.2f} months")
        print(f"    - Root Mean Squared Error (RMSE): {rmse_days:.2f} days | {rmse_months:.2f} months")
        
        # Build the DataFrame
        df_residuals = pd.DataFrame({
            patientID: patient_ids_arr,
            'Risk_Score': risk_scores_filtered,
            'Actual_Days': actual_days,
            'Predicted_Median_Days': pred_days,
            'Absolute_Error_Days': np.abs(actual_days - pred_days),
            'Days_Residual': actual_days - pred_days,
            'Actual_Months': actual_months,
            'Predicted_Median_Months': pred_months,
            'Absolute_Error_Months': np.abs(actual_months - pred_months),
            'Months_Residual': actual_months - pred_months
        }).reset_index(drop=True)

        return df_residuals

    def evaluate_IBS(self, y_train, y_test, survival_functions):
        """
        Compute the Integrated Brier Score (IBS) on the test set to evaluate 
        the model's overall probabilistic survival prediction accuracy.

        The IBS evaluates the squared distance between the predicted survival probability 
        and the actual survival status over a dynamic temporal grid. An IBS < 0.25 
        indicates that the model performs better than an uninformative random guess.

        Args:
            y_train (NumPy structured array): Structured array containing 'Event_Status' 
                and 'Survival_Time' for the training set (used for censoring adjustment).
            y_test (NumPy structured array): Structured array containing 'Event_Status' 
                and 'Survival_Time' for the test set.
            survival_functions (list of sksurv.functions.StepFunction): List of predicted 
                survival curves generated by the model for the test samples.

        Raises:
            ValueError: If the model has not been trained first with fit_crossval().

        Returns:
            float: The calculated Integrated Brier Score (IBS) score.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")

        # --------------------------------------------------------
        #               Integrated Brier Score (IBS)
        # --------------------------------------------------------

        # Temporal grid (intersection of train and test time boundaries)
        min_time = max(y_train["Survival_Time"].min(), y_test["Survival_Time"].min())
        max_time = min(y_train["Survival_Time"].max(), y_test["Survival_Time"].max())
        times_grid = np.linspace(min_time + 1, max_time - 1, num=100)
    
        # Map the StepFunction on the numerical temporal grid
        predictions_matrix = np.asarray([fn(times_grid) for fn in survival_functions])
    
        # Compute IBS
        ibs_score = integrated_brier_score(
            survival_train=y_train,   # to estimate the censoring curve
            survival_test=y_test, 
            estimate=predictions_matrix, 
            times=times_grid
        )
        print(f"Integrated Brier Score (IBS) on Test Set: {ibs_score:.4f}")
        if ibs_score < 0.25:
            print("    -> The model estimates the risk over time better than a random model (0.25).")
        else:
            print("    -> WARNING: High probabilistic error (equal to or worse than a random model).")
         
        return ibs_score
    
    def compute_martingale_and_deviance_residuals(self, X_input, y_input, patient_ids, patientID, 
                                                  predicted_days=None, predicted_months=None, 
                                                  risk_scores=None, hazards=None):
        """
        Compute Martingale and Deviance residuals for each single patient 
        for model validation. Useful for identifying poorly fit outliers.

        Args:
            X_input (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y_input (NumPy structured array): Structured target array (Event_Status, Survival_Time).
            patient_ids (list of str): Patient identifiers.
            patientID (str): Name of the column containing patient identifiers
            predicted_days (numpy.ndarray, optional): Pre-computed median days. If None, uses internal state.
            predicted_months (numpy.ndarray, optional): Pre-computed median months. If None, uses internal state.
            risk_scores (numpy.ndarray, optional): Pre-computed risk scores. If None, they will be computed internally.
            hazards (numpy.ndarray, optional): Pre-computed hazard rates. If None, they will be computed internally.

        Raises:
            ValueError: If the model has not been trained with fit_crossval(),

        Returns:
            pandas.DataFrame: A DataFrame containing Patient IDs, risk indicators, targets, predictions, 
                and both Martingale and Deviance residuals.
        """
        if self.model is None:
            raise ValueError("The model must be trained first with fit_crossval()")
        
        if predicted_days is None or len(predicted_days) != len(X_input):
            predicted_days = getattr(self, 'predicted_medians_d', None)
            
        if predicted_months is None or len(predicted_months) != len(X_input):
            predicted_months = getattr(self, 'predicted_medians_m', None)

        # If still not valid, calculate them internally.
        if predicted_days is None or predicted_months is None or len(predicted_days) != len(X_input):
            predicted_days, predicted_months, _ = self.predict_survival_time(X_input)

        # 1. Extract event status (as 0/1) and survival times
        delta = y_input['Event_Status'].astype(int)
        times = y_input['Survival_Time']

        if risk_scores is None:
            risk_scores = self.compute_risk_scores(X_input)
        risk_scores = np.asarray(risk_scores).ravel()

        if hazards is None:
            hazards = self.compute_hazards(X_input, risk_scores=risk_scores)
        else:
            hazards = np.asarray(hazards).ravel()

        # 2. Extract cumulative hazard functions from the trained model: H_i(t) = -log(S_i(t))
        cum_hazard_funcs = self.model.predict_cumulative_hazard_function(X_input)

        # 3. Evaluate the predicted cumulative hazard exactly at each patient's exit time
        predicted_cum_hazard = []
        for i, fn in enumerate(cum_hazard_funcs):
            patient_time = times[i]
            try:
                val = fn(patient_time)
            except ValueError:
                # Fallback to the last available value if the test time is out of the curve domain calculated on the train set
                val = fn.y[-1]
            predicted_cum_hazard.append(val)

        predicted_cum_hazard = np.array(predicted_cum_hazard)
        # Avoid log(0) or division by zero issues
        predicted_cum_hazard = np.clip(predicted_cum_hazard, 1e-7, None)

        # 4. Compute Martingale Residuals: M_i = delta_i - H_i(t_i)
        martingale_residuals = delta - predicted_cum_hazard

        # 5. Compute Deviance Residuals (Symmetric transformation)
        with np.errstate(invalid='ignore', divide='ignore'):
            adj_term = np.where(delta > 0, delta * np.log(predicted_cum_hazard), 0)    # Corrective term
            deviance_residuals = np.sign(martingale_residuals) * np.sqrt(
                -2 * (martingale_residuals + adj_term)
            )

        # 6. Build the final structured DataFrame
        df_risk_residuals = pd.DataFrame({
            patientID: patient_ids,
            'Risk_Score': risk_scores,
            'Hazard': hazards,
            'Event_Status': y_input['Event_Status'],
            'Survival_Time': times,
            'Predicted_Median_Days': predicted_days,     
            'Predicted_Median_Months': predicted_months,
            'Cumulative_Hazard_Predicted': predicted_cum_hazard,
            'Martingale_Residual': martingale_residuals,
            'Deviance_Residual': deviance_residuals
        }).reset_index(drop=True)

        return df_risk_residuals

# ==========================================
# 3. MODEL CLASS: DEEP COX (DEEPSURV)
# ==========================================

# Create a structure similar to the sksurv StepFunction
# It is used in the Deep Cox model in the method predict_survival_time 
# to evaluate the predicted survival curve at any time point.
class ToolCurve:
    """
    A simple structure to represent a step function for survival curves.
    Mimics the behavior of sksurv.functions.StepFunction.

    Attributes:
        x (numpy.ndarray): Array of survivaltime points where the survival probability changes.
        y (numpy.ndarray): Array of survival probabilities corresponding to each time point in x.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, t):
        """
        Evaluate the survival probability at given time points.

        Args:
            t (float or numpy.ndarray): Time point(s) where the curve is evaluated.

        Returns:
            float or numpy.ndarray: The survival probability at the closest observed time.
        """
        # Return the probability at the time t closest
        idx = np.searchsorted(self.x, t)
        #if idx >= len(self.x): return self.y[-1]
        idx = np.clip(idx, 0, len(self.x) - 1)
        return self.y[idx]

class NegativeLogLikelihoodLoss(nn.Module):
    """
    Loss Function: Negative Log-Likelihood of Cox Partial Likelihood.
    Specifically designed to handle survival data.
    """
    def __init__(self):
        super(NegativeLogLikelihoodLoss, self).__init__()

    def forward(self, risk_scores, events, times):
        """
        Computes the average negative log-partial likelihood for a batch.

        Args:
            risk_scores (torch.Tensor): Predicted log-hazard ratios, shape (batch_size, 1) or (batch_size,).
            events (torch.Tensor): Binary event indicators (1.0 = event, 0.0 = censored), shape (batch_size,).
            times (torch.Tensor): Observed survival times, shape (batch_size,).

        Returns:
            torch.Tensor: Scalar tensor representing the Cox negative log-partial likelihood loss.
        """
        # Order the times in descending order to calculate the risk set (Risk Set)
        # risk_scores shape: (batch_size, 1) or (batch_size,)
        # events shape: (batch_size,)
        # times shape: (batch_size,)
        
        times, indices = torch.sort(times, descending=True)
        events = events[indices]
        risk_scores = risk_scores[indices].squeeze()

        # Calculate the log-sum-exp for the risk set (vectorized)
        # exp_risk = e^(h_i)
        exp_risk = torch.exp(risk_scores)
        # Cumulative sum in reverse order to get the sum of exp_risk for all individuals still at risk
        # cumsum_exp_risk_j = \sum_{j \in R_i} e^(h_j)
        cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
        
        # The partial loss is calculated only for patients who have experienced the event (uncensored)
        log_loss = risk_scores - torch.log(cumsum_exp_risk)
        partial_likelihood = log_loss * events
        
        return -torch.sum(partial_likelihood) / (torch.sum(events) + 1e-8)


class DeepCoxNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture for Deep Cox Regression.
    Features deep learning layers including BatchNorm, ReLU, and Dropout.

    Attributes:
        network (torch.nn.Sequential): The sequential model containing the layers of the MLP.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int, optional): Hidden layer sizes. Defaults to [64, 32].
            dropout_rate (float, optional): Probability of an element to be zeroed in Dropout. Defaults to 0.2.
        """
        super(DeepCoxNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # The output layer has dimension 1 (it corresponds to the linear Risk Score of the classical Cox model)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted log-hazards (risk scores).
        """
        return self.network(x)


class DeepCoxModel:
    """
    It handles the training, optimization and metric extraction of the Deep Cox model.

    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model is trained.
        network (DeepCoxNetwork): The neural network architecture for Deep Cox regression.
        criterion (NegativeLogLikelihoodLoss): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model weights during training.
        unique_times_ (numpy.ndarray or None): Unique survival times from the training data, used for baseline hazard calculation.
        baseline_cumulative_hazard_ (numpy.ndarray or None): Baseline cumulative hazard values corresponding to unique_times_.
        predicted_medians_d (numpy.ndarray or None): 1D array of predicted median survival times in days. 
            Populated after running predict_survival_time().
        predicted_medians_m (numpy.ndarray or None): 1D array of predicted median survival times in months. 
            Populated after running predict_survival_time().
    
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2, lr=1e-4, weight_decay=1e-4, seed=42):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int, optional): Architecture hidden dimensions. Defaults to [64, 32].
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 1e-4.
            weight_decay (float, optional): L2 regularization penalty. Defaults to 1e-4.
            seed (int, optional): Global random seed for reproducibility. Defaults to 42.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = DeepCoxNetwork(input_dim, hidden_dims, dropout_rate).to(self.device)
        self.criterion = NegativeLogLikelihoodLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)

        self.unique_times_ = None
        self.baseline_cumulative_hazard_ = None

        self.predicted_medians_d = None
        self.predicted_medians_m = None
        
    def _prepare_tensors(self, X, y=None):
        """
        Converts NumPy arrays/Structured arrays into PyTorch Tensors placed on the correct device.

        Args:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray, optional): Structured array containing 'Event_Status' and 'Survival_Time'. Defaults to None.

        Returns:
            torch.Tensor or tuple: X tensor, or tuple of (X, events, times) tensors if y is provided.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            # Unpack the sksurv structured array
            events_np = y['Event_Status'].astype(np.float32)
            times_np = y['Survival_Time'].astype(np.float32)
        
            # Create PyTorch tensors 
            events = torch.tensor(events_np, dtype=torch.float32).to(self.device)
            times = torch.tensor(times_np, dtype=torch.float32).to(self.device)
            return X_tensor, events, times
        return X_tensor

    def fit(self, X_train, y_train, epochs=200, batch_size=64, verbose=True, seed=42):
        """
        Trains the Deep Cox network using mini-batch gradient descent and computes
        the baseline cumulative hazard function using the Breslow estimator in the 
        compute_baseline_hazard method.

        Args:
            X_train (numpy.ndarray): Training feature matrix.
            y_train (numpy.ndarray): Structured array containing 'Event_Status' and 'Survival_Time' for training.
            epochs (int, optional): Number of training epochs. Defaults to 200.
            batch_size (int, optional): Mini-batch size. Defaults to 64.
            verbose (bool, optional): If True, prints logs every 10 epochs. Defaults to True.
            seed (int, optional): Reproducibility seed. Defaults to 42.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        X_tensor, events, times = self._prepare_tensors(X_train, y_train)
        dataset = TensorDataset(X_tensor, events, times)
        # Note: Use large batches or shuffle=False if we want to calculate the loss on the entire dataset
        # since partial Cox loss benefits from seeing the entire time spectrum of patients.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.network.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_e, batch_t in dataloader:
                self.optimizer.zero_grad()
                # Risk scores are calculated for each batch of training data to compute the loss (log-hazard)
                risk_scores = self.network(batch_x)
                loss = self.criterion(risk_scores, batch_e, batch_t)
                loss.backward()
                # Risk scores are used to calculate gradients and update the weights of the network through backpropagation
                # After that risk scores are overwritten and so discarded.  
                self.optimizer.step()
                epoch_loss += loss.item()
                
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

        if verbose:
            print("[INFO] Training complete. Computing Breslow baseline hazard...")
        self.compute_baseline_hazard(X_train, y_train, X_tensor=X_tensor)

    def compute_baseline_hazard(self, X_train, y_train, X_tensor=None):
        """
        Calculates the Baseline Cumulative Hazard using the Breslow estimator.
        Necessary for generating individual survival curves.

        Args:
            X_train (numpy.ndarray): Training feature matrix.
            y_train (numpy.ndarray): Structured array containing 'Event_Status' and 'Survival_Time'.
            X_tensor (torch.Tensor, optional): Precomputed feature tensor. Defaults to None.
        """
        self.network.eval()
        with torch.no_grad():
            if X_tensor is None:
                X_tensor = self._prepare_tensors(X_train)
            # The risk score of the network (log-hazard)
            train_risk_scores = self.network(X_tensor).cpu().numpy().squeeze()
            train_hazards = np.exp(train_risk_scores)

        # Extract times and events from the training structured array
        train_times = y_train['Survival_Time']
        train_events = y_train['Event_Status'].astype(int)

        # Order the times of the training set
        sort_idx = np.argsort(train_times)
        train_times = train_times[sort_idx]
        train_events = train_events[sort_idx]
        train_hazards = train_hazards[sort_idx]

        unique_times = np.unique(train_times)
        baseline_hazard = []
        cum_hazard = 0.0

        # Breslow's algorithm
        for t in unique_times:
            # Number of events at time t (numerator of Breslow's estimator)
            events_at_t = np.sum(train_events[train_times == t])
            # Subjects still at risk at time t (survival time >= t)
            risk_set = train_hazards[train_times >= t]
            sum_risk = np.sum(risk_set) if len(risk_set) > 0 else 1.0
            
            cum_hazard += events_at_t / sum_risk
            baseline_hazard.append(cum_hazard)

        self.unique_times_ = unique_times
        self.baseline_cumulative_hazard_ = np.array(baseline_hazard)

    def compute_risk_scores(self, X_input):
        """
        Calculates the Rad-Score (Deep Risk Score) on the test set.

        Args:
            X_input (numpy.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Array of risk scores.
        """
        self.network.eval()   # It deactivates training specific behaviors (e.g., dropout, batch norm).
        with torch.no_grad():     # It deactivates gradient tracking, which reduces memory usage and speeds up computations during inference.
            X_tensor = self._prepare_tensors(X_input)
            risk_scores = self.network(X_tensor).cpu().numpy().reshape(-1)

        return risk_scores
    
    def compute_hazards(self, X_input, risk_scores=None):
        """
        Calculates the hazard rates from the risk scores (e^risk_score).

        Args:
            X_input (numpy.ndarray): Feature matrix.
            risk_scores (numpy.ndarray, optional): Array of risk scores. Defaults to None.

        Returns:
            numpy.ndarray: Array of hazard rates.
        """
        if risk_scores is None:
            risk_scores = self.compute_risk_scores(X_input)
        return np.exp(risk_scores)

    def evaluate_model(self, X_test, y_test, risk_scores=None):
        """
        Calculates the Harrell's C-index on the test set for external validation.

        Args:
            X_test (numpy.ndarray): Feature matrix.
            y_test (numpy.ndarray): Structured array containing 'Event_Status' and 'Survival_Time'.
            risk_scores (numpy.ndarray, optional): Array of risk scores. Defaults to None.

        Returns:
            float: Harrell's C-index.
        """
        if risk_scores is None:
            risk_scores = self.compute_risk_scores(X_test)
        
        # Extract events and times from the sksurv format
        event_status = y_test['Event_Status']
        survival_time = y_test['Survival_Time']
        
        # A major risk corrsponds to a shorter survival time in sksurv
        c_index = concordance_index_censored(event_status, survival_time, risk_scores)[0]
        print(f"Deep-Cox C-index on Test Set: {c_index:.4f}")
        return c_index
    
    def predict_survival_time(self, X_test, risk_scores=None, hazards=None):
        """
        Generates individual step-function survival curves and calculates the median months/days for Deep Cox.

        Args:
            X_test (numpy.ndarray): Feature matrix.
            risk_scores (numpy.ndarray, optional): Array of risk scores. Defaults to None.
            hazards (numpy.ndarray, optional): Array of hazard rates. Defaults to None.

        Raises:
            ValueError: If the baseline hazard is not calculated.

        Returns:
            tuple: Tuple containing predicted medians in days and months, and survival curves.
        """
        if self.baseline_cumulative_hazard_ is None:
            raise ValueError("Baseline hazard not calculated. Execute the fit() method first.")
        
        if hazards is None:
            if risk_scores is None:
                risk_scores = self.compute_risk_scores(X_test)
            hazards = self.compute_hazards(X_test, risk_scores=risk_scores)
        elif risk_scores is None:
            risk_scores = np.log(np.asarray(hazards)).squeeze()
        
        hazards = np.asarray(hazards).ravel()
        risk_scores = np.asarray(risk_scores).ravel()

        predicted_medians_d = []
        survival_curves = [] # Contains dictionaries or arrays with x (times) and y (probabilities)
        average_days_per_month = 30.437

        # Generate the survival curves for each patient in the test set
        for hazard in hazards:
            # S_i(t) = exp(-H_0(t) * exp(risk_score))
            surv_prob = np.exp(-self.baseline_cumulative_hazard_ * hazard)
            
            # Save the curve in a structure similar to the sksurv StepFunction
            fn = ToolCurve(self.unique_times_, surv_prob)
            survival_curves.append(fn)

            # Calculate the median (50%)
            under_50 = np.where(surv_prob <= 0.5)[0]
            if len(under_50) > 0:
                median_time_d = self.unique_times_[under_50[0]]
            else:
                median_time_d = self.unique_times_[-1]
            predicted_medians_d.append(median_time_d)

        self.predicted_medians_d = np.array(predicted_medians_d)
        self.predicted_medians_m = self.predicted_medians_d / average_days_per_month

        return self.predicted_medians_d, self.predicted_medians_m, survival_curves
    
    def compute_residuals_and_metrics(self, y_input, patient_ids, patientID, risk_scores=None, X_input=None):
        """
        Filter out censored patients keeping only those with an observed event, 
        print linear regression metrics (MAE/RMSE), and return a residuals DataFrame.

        Args:
            y_input (pandas.DataFrame): The DataFrame containing the actual survival times and event statuses.
            patient_ids (list): List of patient identifiers.
            patientID (str): Name of the column containing the patient identifiers.
            risk_scores (numpy.ndarray, optional): Array of risk scores for each patient.
            X_input (numpy.ndarray, optional): Array of input features for each patient.

        Raises:
            ValueError: If predict_survival_time() hasn't been run first.
            ValueError: If both risk_scores and X_input are omitted.

        Returns:
            pandas.DataFrame or None: A filtered DataFrame containing residuals for patients with an observed event.
                Returns None if no events are observed in the input data.
        """
        if not hasattr(self, 'predicted_medians_d') or self.predicted_medians_d is None:
            raise ValueError("Predictions missing. Run predict_survival_time() first to populate median times.")

        # Boolean mask to filter only patients with observed events
        event_observed = y_input['Event_Status'] == True
        
        if not np.any(event_observed):
            print("[!] Time Error: No observed events found in the Test Set. Cannot compute MAE/RMSE.")
            return None

        # Extract the filtered data.
        actual_days_all = y_input['Survival_Time']
        actual_months_all = actual_days_all / 30.437
        actual_days = actual_days_all[event_observed]
        actual_months = actual_months_all[event_observed]

        pred_days = self.predicted_medians_d[event_observed]
        pred_months = self.predicted_medians_m[event_observed]

        patient_ids_arr = np.asarray(patient_ids)[event_observed]

        if risk_scores is not None:
            risk_scores_filtered = np.asarray(risk_scores)[event_observed]
        else:
            if X_input is None:
                raise ValueError("Missing data: You must provide either 'risk_scores' or 'X_input' to extract scores.")
            risk_scores = self.compute_risk_scores(X_input)
            risk_scores_filtered = np.asarray(risk_scores)[event_observed]

        # Linear error metrics computation
        mae_days = mean_absolute_error(actual_days, pred_days)
        rmse_days = root_mean_squared_error(actual_days, pred_days)
        mae_months = mean_absolute_error(actual_months, pred_months)
        rmse_months = root_mean_squared_error(actual_months, pred_months)
    
        print(f"Survival Time Error Analysis (N. patients with event = {len(pred_days)}):")
        print(f"    - Mean Absolute Error (MAE):   {mae_days:.2f} days | {mae_months:.2f} months")
        print(f"    - Root Mean Squared Error (RMSE): {rmse_days:.2f} days | {rmse_months:.2f} months")
        
        # Build the Datafram
        df_residuals = pd.DataFrame({
            patientID: patient_ids_arr,
            'Risk_Score': risk_scores_filtered,
            'Actual_Days': actual_days,
            'Predicted_Median_Days': pred_days,
            'Absolute_Error_Days': np.abs(actual_days - pred_days),
            'Days_Residual': actual_days - pred_days,
            'Actual_Months': actual_months,
            'Predicted_Median_Months': pred_months,
            'Absolute_Error_Months': np.abs(actual_months - pred_months),
            'Months_Residual': actual_months - pred_months
        }).reset_index(drop=True)

        return df_residuals

    def evaluate_IBS(self, y_train, y_test, survival_functions):
        """
        Compute the Integrated Brier Score (IBS) on the test set to evaluate 
        the model's overall probabilistic survival prediction accuracy.

        The IBS evaluates the squared distance between the predicted survival probability 
        and the actual survival status over a dynamic temporal grid. An IBS < 0.25 
        indicates that the model performs better than an uninformative random guess.

        Args:
            y_train (NumPy structured array): Structured array containing 'Event_Status' 
                and 'Survival_Time' for the training set (used for censoring adjustment).
            y_test (NumPy structured array): Structured array containing 'Event_Status' 
                and 'Survival_Time' for the test set.
            survival_functions (list of sksurv.functions.StepFunction): List of predicted 
                survival curves generated by the model for the test samples.

        Raises:
            ValueError: If the model has not been trained first with fit().

        Returns:
            float: The calculated Integrated Brier Score (IBS) score
                (IBS < 0.25 indicates informative predictions).
                
        """
        if self.baseline_cumulative_hazard_ is None:
            raise ValueError("The model must be trained first with fit(). ")

        # --------------------------------------------------------
        #               Integrated Brier Score (IBS)
        # --------------------------------------------------------

        # Temporal grid (intersection of train and test time boundaries)
        min_time = max(y_train["Survival_Time"].min(), y_test["Survival_Time"].min())
        max_time = min(y_train["Survival_Time"].max(), y_test["Survival_Time"].max())
        times_grid = np.linspace(min_time + 1, max_time - 1, num=100)
    
        # Map the StepFunction on the numerical temporal grid
        #predictions_matrix = np.asarray([[fn(t) for t in times_grid] for fn in survival_functions])
        predictions_matrix = np.asarray([fn(times_grid) for fn in survival_functions])
    
        # Compute IBS
        ibs_score = integrated_brier_score(
            survival_train=y_train,   # to estimate the censoring curve
            survival_test=y_test, 
            estimate=predictions_matrix, 
            times=times_grid
        )
        print(f"Integrated Brier Score (IBS) on Test Set: {ibs_score:.4f}")
        if ibs_score < 0.25:
            print("    -> The model estimates the risk over time better than a random model (0.25).")
        else:
            print("    -> WARNING: High probabilistic error (equal to or worse than a random model).")
    
        return ibs_score
    
    def compute_martingale_and_deviance_residuals(self, X_input, y_input, patient_ids, patientID, 
                                                  predicted_days=None, predicted_months=None,
                                                  risk_scores=None, hazards=None):
        """
        Computes advanced diagnostic residuals (Martingale and Deviance) for model validation.
        Useful for identifying poorly fit outliers.

        Args:
            X_input (numpy.ndarray): Input feature matrix for evaluation.
            y_input (numpy.ndarray): Structured survival target array.
            patient_ids (list or numpy.ndarray): Identifiers for each patient sample.
            patientID (str): Column name for patient identifiers in the final report.
            predicted_days (numpy.ndarray, optional): Pre-computed median days. If None, uses internal state.
            predicted_months (numpy.ndarray, optional): Pre-computed median months. If None, uses internal state.
            risk_scores (numpy.ndarray, optional): Precalculated risk scores. Defaults to None.
            hazards (numpy.ndarray, optional): Precalculated exponential hazard rates. Defaults to None.

        Raises:
            ValueError: If model is not trained or if predicted medians are missing.

        Returns:
            pandas.DataFrame: Comprehensive diagnostic report containing risks, targets, and calculated residuals.
        """
        if self.baseline_cumulative_hazard_ is None:
            raise ValueError("The model must be trained first with fit(). ")
        
        if predicted_days is None or len(predicted_days) != len(X_input):
            predicted_days = getattr(self, 'predicted_medians_d', None)
            
        if predicted_months is None or len(predicted_months) != len(X_input):
            predicted_months = getattr(self, 'predicted_medians_m', None)

        # If still not valid, calculate them internally.
        if predicted_days is None or predicted_months is None or len(predicted_days) != len(X_input):
            predicted_days, predicted_months, _ = self.predict_survival_time(X_input)

        delta = y_input['Event_Status'].astype(int)
        times = y_input['Survival_Time']
        
        if risk_scores is None:
            risk_scores = self.compute_risk_scores(X_input)
        risk_scores = np.asarray(risk_scores).ravel()
        if hazards is None:
            hazards = self.compute_hazards(X_input, risk_scores=risk_scores)
        hazards = np.asarray(hazards).ravel()

        idx = np.searchsorted(self.unique_times_, times)
        idx = np.clip(idx, 0, len(self.unique_times_) - 1)
    
        # Estract the H0 (Baseline Cumulative Hazard) for all the patients in one go and calculate the cumulative hazard
        H_0 = self.baseline_cumulative_hazard_[idx]
        predicted_cum_hazard = H_0 * hazards
        predicted_cum_hazard = np.clip(predicted_cum_hazard, 1e-7, None)

        # Compute Martingale Residuals
        martingale_residuals = delta - predicted_cum_hazard

        with np.errstate(invalid='ignore', divide='ignore'):
            # since => delta - martingale_residuals = delta - (delta - predicted_cum_hazard) = predicted_cum_hazard
            adj_term = np.where(delta > 0, delta * np.log(predicted_cum_hazard), 0)
            deviance_residuals = np.sign(martingale_residuals) * np.sqrt(
                -2 * (martingale_residuals + adj_term)
            )

        return pd.DataFrame({
            patientID: patient_ids,
            'Risk_Score': risk_scores,
            'Hazard': hazards,
            'Event_Status': y_input['Event_Status'],
            'Survival_Time': times,
            'Predicted_Median_Days': predicted_days,
            'Predicted_Median_Months': predicted_months,
            'Cumulative_Hazard_Predicted': predicted_cum_hazard,
            'Martingale_Residual': martingale_residuals,
            'Deviance_Residual': deviance_residuals
        }).reset_index(drop=True)
        
    
   
   