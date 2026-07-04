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
\lambda(t) = \lim_{\delta\rightarrow0} \frac{\Pr(t\le T<t+\delta|T\ge t)}{\delta}
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

The complete list of requirements for the `nsclc_survival` package is reported in the [requirements.txt](requirements.txt).

> ⚠️ **CRITICAL NOTE ON PYRADIOMICS & NUMPY COMPATIBILITY**
> 
> Due to legacy build constraints in the `pyradiomics` library, running a standard single-step installation (e.g., `pip install -r requirements.txt`, or `pip install -e .`) **will fail** with a `ModuleNotFoundError: No module named 'numpy'`, even if the `numpy` library is present in the requirements file and in setup.py and pyproject.toml.
> 
> `pyradiomics` requires `numpy` to be physically present in the active environment *before* its own metadata generation and compilation processes begin; it cannot resolve `numpy` as a parallel dependency during a bundled installation. 

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

usage: nsclc_survival [-h] [--cv-folds CV_FOLDS] [--epochs EPOCHS]

NSCLC Survival Analysis Pipeline

options:
  -h, --help           show this help message and exit
  --cv-folds CV_FOLDS  Folds for the Lasso Cross-Validation
  --epochs EPOCHS      Epochs of training for Deep Cox
```


## Testing

## How to Cite 

## Author