from pathlib import Path

# --- Base Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- Subdirectories ---
RAW_DATA_PATH = DATA_DIR / "raw_data"
ORGANIZED_DATA_PATH = DATA_DIR / "organized_data"
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_data"
RADIOMICS_FEATURES_PATH = DATA_DIR / "features"
RESULTS_PATH = DATA_DIR / "results"
PLOT_PATH = DATA_DIR / "plots"

CONFIG_PATH = BASE_DIR / "configs"

CLINICAL_FEATURES_PATH = BASE_DIR / "examples"

# --- Config Paths ---
# Centralize the path to the radiomics config file
RADIOMICS_CONFIG_PATH = CONFIG_PATH / "radiomics_config.yaml"

# --- Extracted Features output path ---
RAD_FEATURES_CSV_PATH = RADIOMICS_FEATURES_PATH / "extracted_features.csv"

# --- Clinical features input path ---
CLINICAL_FEATURES_CSV_PATH = CLINICAL_FEATURES_PATH / "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

# --- Download Parameters ---
COLLECTION_NAME = "NSCLC-Radiomics"
N_PATIENTS = 100

# --- Clinical features names columns ---
patientID = "PatientID"
survival_time_col = "Survival.time"
event_status_col = "deadstatus.event"
stage_col = "Overall.Stage"
gender_col = "gender"
histology_col = "Histology"

# --- Mapping for the model ---
# --- Stage Mapping ---
stage_mapping = {'I': 1, 'II': 2, 'IIIa': 3, 'IIIb': 4}
# --- Gender Mapping ---
gender_mapping = {'male': 1, 'female': 0}

# --- Survival curves plots path ---
PLOT_SURVIVAL_CURVES = PLOT_PATH / "survival_curves_comparison.png"
PLOT_DEV_RESIDUALS = PLOT_PATH / "risk_scores_vs_deviance_residuals.png"