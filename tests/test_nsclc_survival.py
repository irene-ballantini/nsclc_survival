#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np
import torch
from sksurv.functions import StepFunction
from nsclc_survival.nsclc_survival import RadiomicsClinicalDataProcessor, LassoCoxModel, ToolCurve, NegativeLogLikelihoodLoss, DeepCoxModel, SurvivalRiskClassifier

# ==========================================
# FIXTURES for MOCK DATA
# ==========================================

@pytest.fixture
def mock_csv_files(tmp_path):
    """
    Create CSV temporary files for radiomics and clinical data.

    Args:
        tmp_path (pathlib.Path): Pytest fixture providing a temporary directory unique 
            to the test invocation.
    """
    # Radiomics Features
    radiomics_data = {
        'PatientID': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'feat_1': [1.2, 2.3, np.nan, 4.1, 5.5],  # Insert a NaN to test the imputer
        'feat_2': [10.0, 20.0, 30.0, 40.0, 50.0]
    }
    radiomics_path = tmp_path / "radiomics.csv"
    pd.DataFrame(radiomics_data).to_csv(radiomics_path, index=False)

    # Clinical Data
    clinical_data = {
        'PatientID': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'Stage': ['I', 'II', 'III', np.nan, 'I'],
        'Gender': ['male', 'female', 'male', 'female', 'male'],
        'Histology': ['Adeno', 'Squamous', 'Adeno', np.nan, 'Squamous'],
        'Survival_Time': [12.5, 24.0, 5.2, 18.3, 45.1 ],
        'Event_Status': [1, 0, 1, 1, 0]
    }
    clinical_path = tmp_path / "clinical.csv"
    pd.DataFrame(clinical_data).to_csv(clinical_path, index=False)

    return str(radiomics_path), str(clinical_path)


@pytest.fixture
def mappings():
    """
    Mapping dictionaries for the tests.

    Returns:
        dict: A dictionary containing stage and gender mapping dictionaries.
    """
    return {
        'stage_mapping': {'I': 1, 'II': 2, 'III': 3},
        'gender_mapping': {'male': 1, 'female': 0}
    }


# ==========================================
# UNIT TESTS
# ==========================================

def test_initialization(mock_csv_files):
    """
    Verify that the initialization sets the paths and attributes correctly.

    Args:
        mock_csv_files (tuple): A tuple containing paths to the mock radiomics and clinical CSV files.
    """
    rad_path, clin_path = mock_csv_files
    processor = RadiomicsClinicalDataProcessor(rad_path, clin_path)
    
    assert processor.radiomics_path == rad_path
    assert processor.clinical_path == clin_path
    assert processor.df_total is None
    assert processor.X_train is None


def test_load_and_merge(mock_csv_files, mappings):
    """
    Verify the correct loading, merging and mapping of columns.

    Args:
        mock_csv_files (tuple): A tuple containing paths to the mock radiomics and clinical CSV files.
        mappings (dict): A dictionary containing stage and gender mapping dictionaries fo the tests.
    """
    rad_path, clin_path = mock_csv_files
    processor = RadiomicsClinicalDataProcessor(rad_path, clin_path)
    
    df = processor.load_and_merge(
        patientID='PatientID',
        stage='Stage',
        gender='Gender',
        histology='Histology',
        stage_mapping=mappings['stage_mapping'],
        gender_mapping=mappings['gender_mapping']
    )
    
    # Verify the merged dataframe
    assert df.shape[0] == 5
    assert 'feat_1' in df.columns
    
    # Verify Stage mapping (NaN should be mapped to the fallback value 3)
    assert df.loc[df['PatientID'] == 'P1', 'Stage'].values[0] == 1
    assert df.loc[df['PatientID'] == 'P4', 'Stage'].values[0] == 3
    
    # Verify Gender mapping
    assert df.loc[df['PatientID'] == 'P1', 'Gender'].values[0] == 1
    assert df.loc[df['PatientID'] == 'P2', 'Gender'].values[0] == 0

    # Verify One-Hot Encoding of Histology (drop_first=True)
    # There should be the columns Histology_Squamous and Histology_Unknown (from NaN)
    assert 'Histology_Squamous' in df.columns
    assert 'Histology_Unknown' in df.columns
    assert 'Histology_Adeno' not in df.columns  


def test_split_and_standardize_raises_error(mock_csv_files):
    """
    Verify that a ValueError is raised if called before load_and_merge.

    Args:
        mock_csv_files (tuple): A tuple containing paths to the mock radiomics and clinical CSV files.
    """
    rad_path, clin_path = mock_csv_files
    processor = RadiomicsClinicalDataProcessor(rad_path, clin_path)
    
    with pytest.raises(ValueError, match="Call load_and_merge\\(\\) before splitting and standardizing."):
        processor.split_and_standardize('PatientID', 'Survival_Time', 'Event_Status')


def test_split_and_standardize_no_duplicates(mock_csv_files, mappings):
    """ 
    Test the split and standardization in the standard case (without duplicates).
    
    Args:
        mock_csv_files (tuple): A tuple containing paths to the mock radiomics and clinical CSV files.
        mappings (dict): A dictionary containing stage and gender mapping dictionaries fo the tests.
    """
    rad_path, clin_path = mock_csv_files
    processor = RadiomicsClinicalDataProcessor(rad_path, clin_path)
    
    processor.load_and_merge(
        patientID='PatientID', stage='Stage', gender='Gender', histology='Histology',
        stage_mapping=mappings['stage_mapping'], gender_mapping=mappings['gender_mapping']
    )
    
    X, y, X_train, X_test, y_train, y_test = processor.split_and_standardize(
        patientID='PatientID', survival_time_col='Survival_Time', event_status_col='Event_Status',
        train_size=0.6, random_seed=42
    )
    
    # Verify the format of the structured array for sksurv (y)
    assert y.dtype.names == ('Event_Status', 'Survival_Time')
    assert y['Event_Status'].dtype == bool
    
    # Verify that the ID and Target columns have been removed from X
    assert 'PatientID' not in processor.feature_names
    assert 'Survival_Time' not in processor.feature_names
    assert 'Event_Status' not in processor.feature_names
    
    # Verify imputation and scaling (X_train should not contain NaN)
    assert not np.isnan(X_train).any() 
    assert not np.isnan(X_test).any()
    # Check that the mean and std are close to 0 and 1 respectively for the training set
    assert -0.5 < np.mean(X_train) < 0.5
    assert 0.5 < np.std(X_train) < 1.5
    # Less strict checks for the test set since it may contain outliers or different distribution
    assert -2 < np.mean(X_test) < 2
    assert 0.2 < np.std(X_test) < 2.5
    
    # Verify correct dimensions based on train_size=0.6 (3 samples train, 2 test)
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert len(processor.patient_ids_train) == 3
    assert len(processor.patient_ids_test) == 2


def test_split_and_standardize_with_duplicates(tmp_path, mappings):
    """
    Verify that in the presence of duplicates, GroupShuffleSplit is used without leakage. 
    
    Args:
        tmp_path (pathlib.Path): The temporary directory path for the test files.
        mappings (dict): A dictionary containing stage and gender mapping dictionaries for the tests.
    """
    # Create a dedicated dataset with duplicate IDs (P1 and P2 appear twice)
    radiomics_data = {
        'PatientID': ['P1', 'P1', 'P2', 'P2', 'P3', 'P4'],
        'feat_1': [1.0, 1.1, 2.0, 2.1, 3.0, 4.0]
    }
    clinical_data = {
        'PatientID': ['P1', 'P1', 'P2', 'P2', 'P3', 'P4'],
        'Survival_Time': [10, 10, 20, 20, 30, 40],
        'Event_Status': [1, 1, 0, 0, 1, 0]
    }
    
    rad_path = tmp_path / "rad_dup.csv"
    clin_path = tmp_path / "clin_dup.csv"
    pd.DataFrame(radiomics_data).to_csv(rad_path, index=False)
    pd.DataFrame(clinical_data).to_csv(clin_path, index=False)
    
    processor = RadiomicsClinicalDataProcessor(str(rad_path), str(clin_path))
    processor.load_and_merge('PatientID', 'Stage', 'Gender', 'Histology', mappings['stage_mapping'], mappings['gender_mapping'])
    
    _, _, X_train, X_test, _, _ = processor.split_and_standardize(
        patientID='PatientID', survival_time_col='Survival_Time', event_status_col='Event_Status',
        train_size=0.5, random_seed=42
    )
    
    # Verify anti-leakage: the same patients should not be in both train and test
    set_train = set(processor.patient_ids_train)
    set_test = set(processor.patient_ids_test)
    
    assert set_train.intersection(set_test) == set()

# ==========================================
# FIXTURES for MOCK DATA for LassoCoxModel
# ==========================================

@pytest.fixture
def mock_features_setup():
    """
    Define the names of the clinical and radiomic features.

    Returns:
        dict: A dictionary containing feature names and the names of the clinical columns.
    """
    feature_names = ['Stage', 'Gender', 'Histology_Squamous', 'Histology_Unknown', 'rad_feat1', 'rad_feat2']
    return {
        'feature_names': feature_names,
        'stage': 'Stage',
        'gender': 'Gender',
        'histology': 'Histology'
    }


@pytest.fixture
def mock_survival_data():
    """
    Generate synthetic X matrices and structured y arrays suitable for sksurv.

    Returns:
        a tuple containing:
            - X_total (np.ndarray): The complete feature matrix.
            - y_total (np.ndarray): The complete structured array for survival data.
            - X_train (np.ndarray): The training feature matrix.
            - X_test (np.ndarray): The test feature matrix.
            - y_train (np.ndarray): The training structured array for survival data.
            - y_test (np.ndarray): The test structured array for survival data.
    """
    np.random.seed(42)
    n_samples = 15  # Minimum number of samples required for a stable 3-fold CV
    n_features = 6
    
    # Matricex X (simulated as standardized)
    X_total = np.random.randn(n_samples, n_features)
    X_train = X_total[:10]
    X_test = X_total[10:]
    
    # Structured Targets y (Event_Status bool, Survival_Time float)
    # Ensure the presence of events (True) and censored observations (False)
    events_total = np.array([True, False, True, True, False, True, False, True, True, False, True, False, True, True, False])
    times_total = np.array([10., 45., 12., 5., 60., 22., 34., 8., 19., 50., 15., 40., 25., 30., 55.])
    
    y_total = np.array(list(zip(events_total, times_total)), dtype=[('Event_Status', '?'), ('Survival_Time', '<f8')])
    y_train = y_total[:10]
    y_test = y_total[10:]
    
    return X_total, y_total, X_train, X_test, y_train, y_test


# ==========================================
# UNIT TESTS (LASSO COX MODEL)
# ==========================================

def test_lasso_cox_initialization(mock_features_setup):
    """
    Verify the correct instantiation of the class and its attributes.

    Args:
        mock_features_setup (dict): A dictionary containing feature names and the names of the clinical columns.
    """
    setup = mock_features_setup
    model_box = LassoCoxModel(setup['feature_names'], setup['stage'], setup['gender'], setup['histology'])
    
    assert model_box.model is None
    assert model_box.best_alpha is None
    assert model_box.stage == 'Stage'
    assert model_box.feature_names[4] == 'rad_feat1'


def test_fit_crossval(mock_features_setup, mock_survival_data):
    """
    Tests the entire training process, GridSearch e Nested CV.
    
    Args:
        mock_features_setup (dict): A dictionary containing feature names and the names of the clinical columns.
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    setup = mock_features_setup
    X_total, y_total, X_train, _, y_train, _ = mock_survival_data
    
    model_box = LassoCoxModel(setup['feature_names'], setup['stage'], setup['gender'], setup['histology'])
    
    # Use a reduced grid and reduce folds/repeats to speed up the unit test
    alpha_grid = {"alphas": [[0.1], [0.5]]}
    
    model_box.fit_crossval(
        X_train=X_train, y_train=y_train, X_total=X_total, y_total=y_total,
        cv=3, n_repeats=1, alpha_param_grid=alpha_grid
    )
    
    assert model_box.best_alpha in [0.1, 0.5]
    assert model_box.model is not None
    # Verify that the final model is ready and configured with the optimal alpha
    assert model_box.model.alphas[0] == model_box.best_alpha


def test_methods_raise_error_before_fit(mock_features_setup, mock_survival_data):
    """
    Verify all computation methods raise an error if the model is not fitted earlier.
    
    Args:
        mock_features_setup (dict): A dictionary containing feature names and the names of the clinical columns.
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    setup = mock_features_setup
    _, _, X_train, X_test, _, y_test = mock_survival_data
    model_box = LassoCoxModel(setup['feature_names'], setup['stage'], setup['gender'], setup['histology'])
    
    with pytest.raises(ValueError, match="The model must be trained first"):
        model_box.evaluate_model(X_test, y_test)
        
    with pytest.raises(ValueError, match="The model must be trained first"):
        model_box.get_selected_features(setup['feature_names'])

    with pytest.raises(ValueError, match="The model must be trained first"):
        model_box.compute_risk_scores(X_train)


@pytest.fixture
def trained_model_box(mock_features_setup, mock_survival_data):
    """
    Support fixture that returns a pre-trained model for testing the first methods.
    
    Args:
        mock_features_setup (dict): A dictionary containing feature names and the names of the clinical columns.
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    setup = mock_features_setup
    X_total, y_total, X_train, _, y_train, _ = mock_survival_data
    model_box = LassoCoxModel(setup['feature_names'], setup['stage'], setup['gender'], setup['histology'])
    
    # Fast training
    alpha_grid = {"alphas": [[0.1]]}
    model_box.fit_crossval(X_train, y_train, X_total, y_total, cv=2, n_repeats=1, alpha_param_grid=alpha_grid)
    return model_box


def test_evaluate_and_features(trained_model_box, mock_survival_data, mock_features_setup):
    """
    Verify the evaluation output (C-index) and feature selection.
    
    Args:
        trained_model_box (LassoCoxModel): A pre-trained instance of the LassoCoxModel.
        mock_survival_data (tuple): A tuple containing the simulated survival data.
        mock_features_setup (dict): A dictionary containing feature names and the names of the clinical columns.
    """
    _, _, _, X_test, _, y_test = mock_survival_data
    setup = mock_features_setup
    
    # Test score (C-index)
    c_index = trained_model_box.evaluate_model(X_test, y_test)
    assert isinstance(c_index, float)
    assert 0.0 <= c_index <= 1.0
    
    # Test feature extraction
    df_features = trained_model_box.get_selected_features(setup['feature_names'])
    assert isinstance(df_features, pd.DataFrame)
    assert 'Feature' in df_features.columns
    assert 'Hazard_Ratio' in df_features.columns
    # Check that features with non-zero coefficients have the correct Hazard Ratio calculated as exp(coef)
    if not df_features.empty:
        first_row = df_features.iloc[0]
        assert np.isclose(first_row['Hazard_Ratio'], np.exp(first_row['Coefficient']))


def test_predictions_and_ibs(trained_model_box, mock_survival_data):
    """
    Test the predictions of median times, risk calculation, and IBS.
    
    Args:
        trained_model_box (LassoCoxModel): A pre-trained instance of the LassoCoxModel.
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    _, _, _, X_test, y_train, y_test = mock_survival_data
    
    # Test prediction curves and median times
    pred_d, pred_m, curves = trained_model_box.predict_survival_time(X_test)
    assert len(pred_d) == len(X_test)
    assert len(pred_m) == len(X_test)
    assert len(curves) == len(X_test)
    assert isinstance(curves[0], StepFunction)
    
    # Test IBS calculation
    ibs = trained_model_box.evaluate_IBS(y_train, y_test, curves)

    assert isinstance(ibs, float)
    assert ibs >= 0.0  # IBS is a squared error, so it cannot be negative


def test_residuals_computations(trained_model_box, mock_survival_data):
    """
    Verify the computation of standard residuals, Martingale and Deviance.
    
    Args:
        trained_model_box (LassoCoxModel): A pre-trained instance of the LassoCoxModel.
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    _, _, _, X_test, _, y_test = mock_survival_data
    patient_ids = [f"Patient_{i}" for i in range(len(X_test))]
    
    # First populate the median times needed for linear residuals
    trained_model_box.predict_survival_time(X_test)
    
    # Test linear residuals (MAE / RMSE)
    df_res = trained_model_box.compute_residuals_and_metrics(
        y_input=y_test, patient_ids=patient_ids, patientID='PatientID', X_input=X_test
    )
    # Must extract only patients with observed EVENT (True)
    n_events = np.sum(y_test['Event_Status'] == True)
    assert len(df_res) == n_events
    assert 'Absolute_Error_Days' in df_res.columns

    # Test advanced residuals (Martingale and Deviance)
    df_advanced_res = trained_model_box.compute_martingale_and_deviance_residuals(
        X_input=X_test, y_input=y_test, patient_ids=patient_ids, patientID='PatientID'
    )
    assert len(df_advanced_res) == len(X_test)  # These are calculated for ALL patients
    assert 'Martingale_Residual' in df_advanced_res.columns
    assert 'Deviance_Residual' in df_advanced_res.columns
    # Verify the vectors don't contain infinities or Nan generated by log(0) or critical divisions
    assert not np.isinf(df_advanced_res['Deviance_Residual']).any()
    assert not df_advanced_res['Deviance_Residual'].isna().any()

# ==========================================
# UNIT TEST PER DEEP COX (DEEPSURV)
# ==========================================

def test_tool_curve_evaluation():
    """
    Tests if ToolCurve class correctly interpolates temporal points.
    """
    times = np.array([10, 20, 30, 40])
    probabilities = np.array([1.0, 0.8, 0.5, 0.2])
    
    curve = ToolCurve(times, probabilities)
    
    # Test exact match
    assert curve(10) == 1.0
    # Test interpolation/step function
    assert curve(25) == 0.5  # Between 20 and 30, the next index is taken via searchsorted/clip
    # Test clip upper bound
    assert curve(50) == 0.2


def test_deep_cox_loss_computation():
    """
    Verify that the Deep Cox loss does not return NaN and decreases if the order is correct.
    """
    loss_fn = NegativeLogLikelihoodLoss()
    
    # 3 fake patients sorted by descending time (as required by the internal calculation)
    times = torch.tensor([120.0, 80.0, 30.0], dtype=torch.float32)
    events = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32) # Fake patient 2 is censored
    risk_scores = torch.tensor([0.5, -0.2, 1.1], dtype=torch.float32)
    
    loss = loss_fn(risk_scores, events, times)
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0  # The Negative Log-Likelihood Loss must be a positive quantity

def test_deep_cox_prepare_tensors(mock_survival_data):
    """
    Verify if the internal tensor preparation method correctly converts sksurv/numpy data 
    into valid PyTorch objects.

    Args:
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    _, _, X_train, _, y_train, _ = mock_survival_data
    
    model = DeepCoxModel(input_dim=X_train.shape[1], hidden_dims=[8, 4])
    
    X_tensor, events, times = model._prepare_tensors(X_train, y_train)
    assert X_tensor.shape[0] == len(y_train)
    assert X_tensor.shape[1] == X_train.shape[1]
    assert isinstance(X_tensor, torch.Tensor)
    assert isinstance(events, torch.Tensor)


def test_deep_cox_fit_and_prediction_pipeline(mock_survival_data):
    """
    Verify the entire DeepCox pipeline: fit, baseline hazard, predictions and curves.

    Args:
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    _, _, X_train, X_test, y_train, _ = mock_survival_data
    input_dim = X_train.shape[1]
    
    # Model initialization
    model = DeepCoxModel(input_dim=input_dim, hidden_dims=[8, 4])
    
    # Check that the baseline hazard is empty before fitting
    assert model.baseline_cumulative_hazard_ is None
    
    # Fast training (few epochs for quick testing)
    model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=False)
    
    # Verify that Breslow has populated the attributes
    assert model.baseline_cumulative_hazard_ is not None
    assert model.unique_times_ is not None
    assert len(model.baseline_cumulative_hazard_) == len(model.unique_times_)
    # The Baseline hazard must be monotonic increasing
    assert np.all(np.diff(model.baseline_cumulative_hazard_) >= 0)
    
    # Verify computation of Risk Scores and Hazards
    risk_scores = model.compute_risk_scores(X_test)
    hazards = model.compute_hazards(X_test, risk_scores=risk_scores)
    assert len(risk_scores) == len(X_test)
    assert np.allclose(hazards, np.exp(risk_scores))
    
    # Verify prediction of survival curves and median times
    pred_d, pred_m, curves = model.predict_survival_time(X_test)
    assert len(pred_d) == len(X_test)
    assert len(curves) == len(X_test)
    assert isinstance(curves[0], ToolCurve)
    
    # Verify the correct mathematical conversion from days to months
    assert np.allclose(pred_m, pred_d / 30.437)


def test_deep_cox_residuals_and_diagnostics(mock_survival_data):
    """
    Tests the extraction of standard (MAE) and advanced (Martingale and Deviance) residuals.

    Args:
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    _, _, X_train, X_test, y_train, y_test  = mock_survival_data
    input_dim = X_train.shape[1]

    patient_ids_test = [f"PATIENT_{i}" for i in range(len(X_test))]
    
    model = DeepCoxModel(input_dim=input_dim, hidden_dims=[8, 4])
    model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=False)
    
    # Prepare the state for residuals by running predict_survival_time first
    model.predict_survival_time(X_test)
    
    # Convert y_test in DataFrame as required by compute_residuals_and_metrics
    y_test_df = pd.DataFrame({
        'Event_Status': y_test['Event_Status'],
        'Survival_Time': y_test['Survival_Time']
    })
    
    # Test standard residuals (filtered only on observed events)
    df_res = model.compute_residuals_and_metrics(y_test_df, patient_ids_test, patientID="PatientID", X_input=X_test)
    if df_res is not None:  # Pass if there are events in the mock set
        actual_days = y_test_df['Survival_Time'][y_test_df['Event_Status'] == True]
        pred_d = model.predicted_medians_d[y_test_df['Event_Status'] == True]
        assert len(actual_days) == len(pred_d)
        assert len(actual_days) > 0

        assert isinstance(df_res, pd.DataFrame)
        assert "Absolute_Error_Days" in df_res.columns
        assert "Days_Residual" in df_res.columns
    
    # Test diagnostic residuals (Martingale and Deviance for ALL patients)
    df_diagnostics = model.compute_martingale_and_deviance_residuals(
        X_input=X_test, y_input=y_test, patient_ids=patient_ids_test, patientID="PatientID"
    )
    
    assert isinstance(df_diagnostics, pd.DataFrame)
    assert len(df_diagnostics) == len(X_test)
    assert "Martingale_Residual" in df_diagnostics.columns
    assert "Deviance_Residual" in df_diagnostics.columns
    
    # Fundamental theoretical check: Martingale residuals are upper bounded by 1
    assert np.all(df_diagnostics["Martingale_Residual"] <= 1.0)
    assert -2 < np.mean(df_diagnostics["Deviance_Residual"]) < 2  # Mean should be around 0

# ==========================================
# UNIT TESTS (RISK CLASSIFIER)
# ==========================================

def test_survival_risk_classifier_pipeline(mock_survival_data):
    """
    Verify the SurvivalRiskClassifier pipeline:
    threshold calculation, classification, log-rank computation, reports and matrices.

    Args:
        mock_survival_data (tuple): A tuple containing the simulated survival data.
    """
    _, _, X_train, X_test, y_train, y_test = mock_survival_data
    input_dim = X_train.shape[1]
    
    # Prepare the model (DeepCoxModel)
    model = DeepCoxModel(input_dim=input_dim, hidden_dims=[8, 4])
    model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=False)
    
    # Prepare target DataFrames for the classifer
    y_train_df = pd.DataFrame({
        'Event_Status': y_train['Event_Status'],
        'Survival_Time': y_train['Survival_Time']
    })
    y_test_df = pd.DataFrame({
        'Event_Status': y_test['Event_Status'],
        'Survival_Time': y_test['Survival_Time']
    })
    patient_ids_test = [f"PATIENT_{i}" for i in range(len(X_test))]
    
    # Initialize the risk classifier
    classifier = SurvivalRiskClassifier(trained_model=model)
    assert classifier.threshold_ is None
    
    # Test fit_threshold
    classifier.fit_threshold(X_train=X_train, verbose=False)
    assert classifier.threshold_ is not None
    assert isinstance(classifier.threshold_, (float, np.floating))
    
    # Test predict_risk_class
    y_pred_class = classifier.predict_risk_class(X_input=X_test)
    assert len(y_pred_class) == len(X_test)
    assert np.all((y_pred_class == 0) | (y_pred_class == 1)) # Verify that it is a binary array
    
    # Test evaluate_stratification (Log-Rank Test)
    # Mock the logrank_test function from lifelines if we don't want to depend on stochastic convergence
    p_value = classifier.evaluate_stratification(y_test_df, y_pred_class, title_suffix="Test_Run")
    assert isinstance(p_value, float)
    assert 0.0 <= p_value <= 1.0
    
    # Test compute_classification_report
    conf_matrix = classifier.compute_classification_report(y_test_df, y_train_df, y_pred_class)
    assert isinstance(conf_matrix, np.ndarray)
    assert conf_matrix.shape == (2, 2) # Binary confusion matrix
    
    # Test generate_prediction_report
    pred_days, _, _ = model.predict_survival_time(X_test)
    risk_scores_test = model.compute_risk_scores(X_test)
    
    df_report = classifier.generate_prediction_report(
        patient_ids=patient_ids_test,
        y_test=y_test_df,
        predicted_time=pred_days,
        y_pred_class=y_pred_class,
        risk_scores_test=risk_scores_test
    )
    
    assert isinstance(df_report, pd.DataFrame)
    assert len(df_report) == len(X_test)
    assert df_report['Risk_Score'].iloc[0] >= df_report['Risk_Score'].iloc[-1] # Check ordering in descending order

def test_survival_risk_classifier_significant_pvalue(mock_survival_data, capsys, mocker):
    """
    Verify that evaluate_stratification returns the correct p-value
    and prints the success message when the p-value is significant (< 0.05).

    Args:
        mock_survival_data (tuple): A tuple containing the simulated survival data.
        capsys: Pytest fixture to capture stdout and stderr output.
        mocker: Pytest-mock fixture for dependency injection.
    """
    _, _, X_train, X_test, y_train, y_test = mock_survival_data
    input_dim = X_train.shape[1]
    
    # Prepare the model (DeepCoxModel)
    model = DeepCoxModel(input_dim=input_dim, hidden_dims=[8, 4])
    model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=False)
    
    # Prepare target DataFrames for the classifer
    y_test_df = pd.DataFrame({
        'Event_Status': y_test['Event_Status'],
        'Survival_Time': y_test['Survival_Time']
    })

    classifier = SurvivalRiskClassifier(trained_model=model)
    classifier.fit_threshold(X_train=X_train, verbose=False)
    y_pred_class = classifier.predict_risk_class(X_input=X_test)
    
    # Create the mock for logrank_test to return a fixed significant p-value
    fake_result = mocker.MagicMock()
    fake_result.p_value = 0.01

    # Execute the code inside the patch context
    mocker.patch('nsclc_survival.nsclc_survival.logrank_test', return_value=fake_result)
    p_value = classifier.evaluate_stratification(y_test_df, y_pred_class, title_suffix="Test_Run")
        
    # Check the printing
    captured = capsys.readouterr()

    expected_message = "-> The classification separated the patients in a statistically significant way (p < 0.05)."
    assert expected_message in captured.out
        
    assert p_value == 0.01

def test_survival_risk_classifier_exceptions(mock_survival_data):
    """
    Verify the SurvivalRiskClassifier raises correct exceptions (ValueError)
    when data is missing or execution order is incorrect.
    """
    _, _, X_train, X_test, _, _ = mock_survival_data
    model = DeepCoxModel(input_dim=X_train.shape[1], hidden_dims=[8, 4])
    
    classifier = SurvivalRiskClassifier(trained_model=model)
    
    # Exception 1: Call fit_threshold without arguments
    with pytest.raises(ValueError, match="Provide 'X_train' or 'risk_scores_train' first."):
        classifier.fit_threshold()
        
    # Exception 2: Call predict_risk_class before computing the threshold (threshold_)
    with pytest.raises(ValueError, match="Threshold not computed. Execute fit_threshold"):
        classifier.predict_risk_class(X_input=X_test)
        
    # Calculate the threshold to unlock the second check on predict_risk_class
    classifier.fit_threshold(X_train=X_train, verbose=False)
    
    # Exception 3: Call predict_risk_class without passing any matrix or score
    with pytest.raises(ValueError, match="Provide 'X_input' or 'risk_scores' first."):
        classifier.predict_risk_class()