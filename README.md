# nsclc_survival
NSCLC Radiomics: Survival Time prediction using CT-extracted features and clinical data.

## Project Overview
Survival Analysis plays a pivotal role in the medical field, especially in oncology, where predicting disease progression and survival outcomes becomes essential for cancer prognosis and treatment planning. 

The core objective of this study is to predict patient survival outcomes and stratify individual risk using the public **NSCLC-Radiomics dataset**, also investigating which radiomic and clinical features most significantly affect patient survival. 

By extracting radiomic and clinical features, the pipeline handles data censorship and implements two primary methodological approaches:
1. **Standard Lasso-Cox Proportional Hazards Model:** Used as a semi-parametric linear baseline.
2. **Deep Cox Proportional Hazards Network (DeepSurv):** A deep learning-based framework designed to capture complex, non-linear feature interactions.

These approaches are statistical methods designed to analyze the time until a specific event occurs (e.g., the patient death). They are built to handle censored data, so situations where the event has not occurred before the end of the study, or the subject is lost to follow-up.

A classification of the patients through a risk stratification is also performed to divide the dataset into High and Low Risk classes, containing patients with an expected survival below and above the median respectively.

For the evaluation of the forementioned classification the following Non-Parametric Methods are adopted:
1. **Kaplan-Meier Method**, which estimates the unadjusted probability of surviving beyond a certain time point; particularly, a Kaplan-Meier curve shows the estimated survival function by plotting estimated survival probabilities against time. 
2. **Log-Rank Test**, which tests the null hypothesis that there is no difference in the probability of an event at any time point across the different curves, evaluating through the computation of the p-value whether the distance between the survival curves of two or more different groups is statistically significant or simply due to chance.

## Survival Data
Survival data is comprised of three elements: baseline data $x$, an event time $T$, and an event indicator $E$. The time $T$ corresponds to the time elapsed between the time in which the baseline data was collected and the time of the event occurring (when $E=1$), or the time of the last contact with the patient (when $E=0$). 

Two fundalmental elements for the survival analysis are:
1. **Survival Function**: it describes the probability that an individual
survives past a specified time point $t$ and is denoted by: 

$$
S(t)=\Pr(T>t)
$$

2. **Hazard Function**: it describes  the instantaneous hazard rate over time, which represents the rate of occurence of the event during an infinitesimally small time interval. Its value is not a probability, but an indicator of the risk of experiencing the event. It  is linked to the probability of an individual dying at time $t$ given that he or she has survived up to that point, and can be defined as follows: 

$$
\lambda(t) = \lim_{\delta \to 0} \frac{1}{\delta} \Pr(t \le T < t + \delta \mid T \ge t)
$$ 

## Proportional Hazards Models
Proportional hazards models are common methods for modeling an individual’s survival given their baseline data $x$. 

For these models the corresponding hazard function takes the form: 

$$
\lambda(t|x) = \lambda_0(t)\cdot e ^{h(x)}
$$

where $\lambda_0(t)$ is the baseline hazard function, and $h(x)$ is the risk function. 

In **Cox Proportional Hazards model** this hazard function becomes: 

$$
\lambda(t|x)=\lambda_0(t)\cdot e^{\beta_1 x_1+...+\beta_p x_p}
$$

where $\lambda(t|x)$ is the hazard at time $t$, $x_1,...,x_p$ are the predictors, $\lambda_0(t)$ is the baseline hazard function common to all patients, and $\beta_1,...,\beta_p$ are the model parameters describing the effect of the predictors on the overall hazard.  Under this formulation, each subject's individual hazard function is obtained by multiplying the common baseline hazard by the subject-specific factor $e^{h(x)}$, where $h(x)=\beta_1 x_1+...+\beta_p x_p$ represents the linear risk function (or log-hazard). The quantity $e^{h(x)}$ is therefore a relative risk multiplier.

The ratio of the hazard rates between different patients is defined as the hazard ratio (HR). An HR greater than 1 indicates that the event is more likely to occur (increased risk), while an HR less than 1 indicates the event is less likely to occur (decreased risk). An HR of exactly 1 signifies that the predictor has no effect on the hazard of the event. 

To perform Cox regression, the parameters $\beta$ are tuned to optimize the Cox partial likelihood, which computes the probability at each event time $T_i$ that the event occurred to individual $i$, given the set of individuals who are still at risk at that same time $T_i$. It can can be defined as: 

$$
L_c(\beta)=\prod_{i:E_i=1}\frac{\exp(h_\beta(x_i))}{\sum_{j\in \mathcal{R}(T_i)} \exp(h_\beta(x_j))}
$$

where the values $T_i$, $E_i$, and $x_i$ are the respective event time, event indicator, and baseline data for the $i^{th}$ observation. The risk set $\mathcal{R}(t)={i:T_i \ge t}$ represents the set of patients still at risk of death at time $t$. There the notation $h_\beta(x)$ has been used instead of $h(x)$ to make clear the dependence on the parameters $\beta$.

However, classic Cox PH model may be too simplistic for fitting complex, real-world biological datasets, and for this reason the **Deep Cox Proportional Hazards Network (DeepSurv)** was implemented, extending the traditional Cox PH model by exploiting neural networks. DeepSurv is a multi-layer perceptron where a deep architecture and modern deep learning techniques - such as Weight Decay Regularization, Rectified Linear Units (ReLU), Batch Normalization, Dropout - ensure stable training, accelerate convergence, and prevent overfitting. The output of the network is a single node, which estimates the risk function $h_\theta(x)$ parameterized by the weights of the network $\theta$. The loss function is set to be the negative log partial likelihood:

$$
\ell(\theta) := - \sum_{i:E_i=1}\bigg(h_\theta(x_i)-\log \sum_{j \in \mathcal{R}(T_i)}e^{h_\theta(x_j)}\bigg)
$$

DeepSurv's major strenght is the ability to generate personalized treatment recommendations. 

## Installation

Python version supported : ![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-green)

### Prerequisites

Before installing the package, please ensure your system satisfies the following requirements:

1. **C++ Compiler**: Due to the underlying C/C++ extensions in `pyradiomics` and `scikit-survival`, a C++ compiler must be present on the system.
2. **Python Version**: This project requires **Python 3.8, 3.9 or 3.10** (3.9+ recommended due to dependencies such as `scikit-survival` and `pyradiomics`). Also for this reason it is recommended to use a virtual environment with a supported Python version.

The complete list of requirements for the `nsclc_survival` package is reported in the [requirements.txt](https://github.com/irene-ballantini/nsclc_survival/blob/main/requirements.txt).

| :warning: CRITICAL NOTE ON PYRADIOMICS & NUMPY COMPATIBILITY |
|:------------------|
| Due to legacy build constraints in the `pyradiomics` library, running a standard single-step installation (e.g., `pip install -r requirements.txt`, or `pip install -e .`) **will fail** with a `ModuleNotFoundError: No module named 'numpy'`, even if the `numpy` library is present in the requirements file and in setup.py and pyproject.toml. `pyradiomics` requires `numpy` to be physically present in the active environment *before* its own metadata generation and compilation processes begin; it cannot resolve `numpy` as a parallel dependency during a bundled installation. |

---
### Setup Instruction (Important)
To ensure a smooth setup and install `nsclc_survival` package in `Python`, install the dependencies sequentially by following these steps:

1. Clone the repository
   ```
   git clone https://github.com/irene-ballantini/nsclc_survival
   cd nsclc_survival
   ```
2. Create and activate your virtual environment (venv/conda) with one of the supported Python versions inside the `nsclc_survival` folder. 
If you use venv, you can safely create the environment folder inside the project root; the .gitignore file is already pre-configured to ignore common names such as venv/, .venv/, env/, or nsclc_env/.

3. Install the dependencies
   ```
   pip install numpy
   pip install --editable . --no-build-isolation
   ```

## Usage
Once the installation is complete, you can run the NSCLC Survival Analysis pipeline directly from your terminal. 

The project includes a Command Line Interface (CLI) that allows you to customize the execution parameters without modifying the source code.

### Running the Pipeline

You can launch the program in two equivalent ways (ensure your virtual environment is active):

1. Using the package shortcut
   ```
   nsclc_survival
   ```

2. Or using the standard Python module syntax
   ```
   python -m nsclc_survival
   ```
### Command Line Interface (CLI) Options
You can append optional arguments to the command to modify the pipeline parameters. The full list of available flags for the customization of the command line can be obtained by calling:
```bash
$ nsclc_survival --help

usage: nsclc_survival [-h] [--n-patients N_PATIENTS] [--cv-folds CV_FOLDS] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--hidden-dims HIDDEN_DIMS [HIDDEN_DIMS ...]] [-v]

NSCLC Survival Analysis Pipeline: Survival Time prediction using CT-extracted features and clinical data.

options:
  -h, --help            show this help message and exit
  --n-patients N_PATIENTS
                        Number of patients to download (default from settings: 100)
  --cv-folds CV_FOLDS   Folds for the Lasso Cross-Validation in Cox model. Default to 5.
  --epochs EPOCHS       Epochs of training for Deep Cox
  --batch-size BATCH_SIZE
                        Batch size of training for Deep Cox
  --hidden-dims HIDDEN_DIMS [HIDDEN_DIMS ...]
                        Hidden dimensions for the Deep Cox neural network (e.g., --hidden-dims 128 64 32)
  -v, --version         Show the current package version and exit
```

## Testing

A full set of testing functions is provided in the [tests](tests) directory.

The tests are performed using the `pytest` python package. You can run the full list of tests with:

```
python -m pytest tests --cov=nsclc_survival --cov-config=.coveragerc
```
in the root directory.

## Table of Contents

Description of the folders related to the `Python` version.

| **Directory**                                                                                | **Description**                                                                                       |
|:---------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
|[examples](https://github.com/irene-ballantini/nsclc_survival/tree/main/examples)             | Examples of CSV files containing the clinical features to merge with the extracted radiomics features.|
|[configs](https://github.com/irene-ballantini/nsclc_survival/tree/main/configs)               | Configuration files (YAML) for `pyradiomics` feature extraction.                                             |
|[nsclc_survival](https://github.com/irene-ballantini/nsclc_survival/tree/main/nsclc_survival) | List of `Python` scripts for the `nsclc_survival` pipeline.                                           | 

Below there's an overview of the project's directory tree created once the package is ran.

```text
nsclc_survival
├── configs/
│   └── radiomics_config.yaml       # Configuration for pyradiomics feature extraction
├── examples/
│   └── NSCLC-Radiomics-Lung1...csv # Example clinical features dataset
├── nsclc_survival/
│   ├── __init__.py
│   ├── settings.py                 # Global settings and path configurations
│   └── ...                         # Core Python package source code
└── data/                           # Created automatically by the pipeline
    ├── raw_data/                   # Downloaded raw imaging data (CT, RTSTRUCT)
    ├── organized_data/             # Sorted/organized data
    ├── preprocessed_data/          # Processed images ready for extraction
    ├── features/                   # Contains 'extracted_features.csv'
    ├── results/                    # Model outputs and metrics
    └── plots/                      # Generated survival curves and residual plots
```

## Configuration and Customization
In [nsclc_survival](https://github.com/irene-ballantini/nsclc_survival/tree/main/nsclc_survival) there's the [settings.py](https://github.com/irene-ballantini/nsclc_survival/blob/main/nsclc_survival/settings.py) file which is a configuration file to configure and centralize global constants, download parameters, and file directories and paths. By modifying this file, you can customize:
* **Download parameters**: (e.g., N_PATIENTS to change the dataset size). The number of patient to download can also be changed via command line.
* **Dataset Structure**: Column names of the clinical CSV files and survival mappings (e.g., stage_mapping for the overall tumor stage).
* **Saving directories**: Saving directories and filenames.

> [!WARNING]
> **DO NOT MODIFY** the paths listed under the `# --- Subdirectories --- ` section. These constants define the internal package structure, allowing the pipeline to automatically create and manage the `data/` directory tree.

> [!NOTE]
> You can also change the `COLLECTION_NAME` constant to download a different dataset from **The Cancer Imaging Archive (TCIA)**, provided that its structure matches or is highly similar to the `NSCLC-Radiomics` dataset.

## Pipeline Workflow
The dataset undergoes a structured pipeline, moving through the following stages:

1. **`_download_data.py`**: Downloads and stores the raw data in the `raw_data/` folder. Files are grouped into separate folders named after their **Unique Identifiers (UID)** - in the DICOM standard, a UID is a unique, globally standardized numeric string used to unambiguously identify medical imaging objects. At this stage, there is no human-readable distinction between patient IDs and modalities (`CT`, `RTSTRUCT`, `SEG`).
2. **`_organize_data.py`**: Reorganizes the raw data into the `organized_data/` folder using patient-specific subdirectories. Each folder contains the `CT` series along with its matching `RTSTRUCT` and `SEG` files. Once this reorganization step is completed successfully, the `raw_data/` folder is automatically removed to save disk space.
3. **`preprocessing.py`**: Converts the organized DICOM data into **NIfTI format (`.nii.gz`)** and saves them in `preprocessed_data/<PatientID>/`. Specifically, this step:
   * Extracts the primary tumor mask (`GTV-1` ROI) from the `RTSTRUCT` vector coordinates and converts it into a binary spatial volume using `rt_utils` and `SimpleITK`.
   * Resamples both the `CT` image (using BSpline interpolation) and the tumor mask (using Nearest Neighbor interpolation) to a **1.0mm isotropic spacing** to ensure spatial consistency for radiomics.
   * Outputs two standardized files per patient: `image.nii.gz` (the resampled CT) and `label.nii.gz` (the resampled tumor mask).
4. **`feature_extraction.py`**: Extracts radiomics features from the preprocessed NIfTI files (`image.nii.gz` and `label.nii.gz`) using the **PyRadiomics** framework. The feature extraction parameters are specified in [radiomics_config.yaml](https://github.com/irene-ballantini/nsclc_survival/blob/main/configs/radiomics_config.yaml). In the end, all results are merged into a single structured list of dictionaries, map-indexed by `PatientID`, ready to be exported as a clean CSV dataset to the `RAD_FEATURES_CSV_PATH` directory defined in [settings.py](https://github.com/irene-ballantini/nsclc_survival/blob/main/nsclc_survival/settings.py).
5. **`nsclc_survival.py`**: Implements the survival modeling framework. Specifically, this script handles:
   * **Data Preparation**: Splits the dataset into training and test sets and performs feature standardization.
   * **Model Training**: Trains both a clinical-radiomics **Standard Lasso-Cox Model** (for baseline statistical modeling) and a **Deep Cox Model** (for non-linear deep learning survival analysis).
   * **Evaluation & Risk Stratification**: Computes risk scores to classify patients into risk groups and generates evaluation outputs (e.g., Kaplan-Meier survival curves and deviance residual plots) saved in the `results/` and `plots/` directories.

## How to Cite 

## Author