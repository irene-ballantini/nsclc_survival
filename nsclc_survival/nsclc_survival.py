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
            survival_time_col (str): Column name representing the time-to-event (Death/Event) or 
                last follow-up time(censoring).
            event_status_col (str): Column name representing the binary event occurrence 
                (e.g., 1 for death/recurrence, 0 for censored (patient still alive at last follow-up)).
            train_size (float): Proportion of the dataset to include in the train split. Defaults to 0.8.
            random_seed (int): Random seed for reproducibility. Defaults to 42.

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
        #self.feature_names = X.columns
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
    while also providing a measure of model performance through the C-index.  

    Attributes:
        model (CoxnetSurvivalAnalysis or None): The final estimator optimized
            via Cross-Validation. Initially set to None.
        best_alpha (float or None): The optimal alpha penalty parameter found 
            during optimization. Initially set to None.
    """
    def __init__(self, feature_names, stage, gender, histology):
        self.model = None
        self.best_alpha = None
        self.feature_names = feature_names
        self.stage = stage
        self.gender = gender
        self.histology = histology

    def fit_crossval(self, X_train, y_train, cv=5, n_repeats=3, alpha_param_grid=None):     
        print("Starting optimized research through GridSearchCV...")
        
        # 1. Use ElasticNet
        l1_ratio_chosen = 1.0
        
        # 2. Define a logarithmic grid for alphas.
        # Avoid too big values that brings everything to 0,
        # and avoid very small values (e.g. < 1e-4) that cause the ArithmeticError because of the weights too large.
        if alpha_param_grid is None:
            alpha_param_grid= {
                "alphas": [[a] for a in np.logspace(-1.0, -0.1, num=20)]
            }
        
        # Crate an array of weights for the penalty as long as the column in X 
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
        
        # 3. Train the final model
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
            feature_names (list of str or pandas.Index): Full list containing the names of all original features.

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

        # Create a DataFrame with Feature, Coefficient and Hazard Ratio
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
        
    