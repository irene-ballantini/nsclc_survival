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
survives past a specified time point $t$ and is denoted by: $$S(t)=\Pr(T>t)$$
2. **Hazard Function**: it describes  the instantaneous hazard rate over time, which represents the rate of occurence of the event during an infinitesimally small time interval. Its value is not a probability, but an indicator of the risk of experiencing the event. It  is linked to the probability of an individual dying at time $t$ given that he or she
has survived up to that point, and can be defined as follows: $$\lambda(t) = \lim_{\delta\rightarrow0} \frac{\Pr(t\le T<t+\delta|T\ge t)}{\delta}$$ 

## Proportional Hazards Models
Proportional hazards models are common methods for modeling an individual’s survival given their baseline data $x$. 

For these models the corresponding hazard function takes the form: $$\lambda(t|x) = \lambda_0(t)\cdot e ^{h(x)}$$
where $\lambda_0(t)$ is the baseline hazard function, and $h(x)$ is the risk function. 

In **Cox Proportional Hazards model** this hazard function becomes: $$\lambda(t|x)=\lambda_0(t)\cdot e^{\beta_1 x_1+...+\beta_p x_p}$$
where $\lambda(t|x)$ is the hazard at time $t$, $x_1,...,x_p$ are the predictors, $\lambda_0(t)$ is the baseline hazard function common to all patients, and $\beta_1,...,\beta_p$ are the model parameters describing the effect of the predictors on the overall hazard.  Under this formulation, each subject's individual hazard function is obtained by multiplying the common baseline hazard by the subject-specific factor $e^{h(x)}$, where $h(x)=\beta_1 x_1+...+\beta_p x_p$ represents the linear risk function (or log-hazard). The quantity $e^{h(x)}$ is therefore a relative risk multiplier.

The ratio of the hazard rates between different patients is defined as the hazard ratio (HR). An HR greater than 1 indicates that the event is more likely to occur (increased risk), while an HR less than 1 indicates the event is less likely to occur (decreased risk). An HR of exactly 1 signifies that the predictor has no effect on the hazard of the event. 

To perform Cox regression, the parameters $\beta$ are tuned to optimize the Cox partial likelihood, which computes the probability at each event time $T_i$ that the event occurred to individual $i$, given the set of individuals who are still at risk at that same time $T_i$. It can can be defined as: $$L_c(\beta)=\prod_{i:E_i=1}\frac{\exp(h_\beta(x_i))}{\sum_{j\in \mathcal{R}(T_i)} \exp(h_\beta(x_j))}$$
where the values $T_i$, $E_i$, and $x_i$ are the respective event time, event indicator, and baseline data for the $i^{th}$ observation. The risk set $\mathcal{R}(t)={i:T_i \ge t}$ represents the set of patients still at risk of death at time $t$. There the notation $h_\beta(x)$ has been used instead of $h(x)$ to make clear the dependence on the parameters $\beta$.

However, classic Cox PH model may be too simplistic for fitting complex, real-world biological datasets, and for this reason the **Deep Cox Proportional Hazards Network (DeepSurv)** was implemented, extending the traditional Cox PH model by exploiting neural networks. DeepSurv is a multi-layer perceptron where a deep architecture and modern deep learning techniques - such as Weight Decay Regularization, Rectified Linear Units (ReLU), Batch Normalization, Dropout - ensure stable training, accelerate convergence, and prevent overfitting. The output of the network is a single node, which estimates the risk function $h_\theta(x)$ parameterized by the weights of the network $\theta$. The loss function is set to be the negative log partial likelihood: 
$$\ell(\theta) := - \sum_{i:E_i=1}\bigg(h_\theta(x_i)-\log \sum_{j \in \mathcal{R}(T_i)}e^{h_\theta(x_j)}\bigg)$$

DeepSurv's major strenght is the ability to generate personalized treatment recommendations. 




## Prerequisites

The complete list of requirements for the `nsclc_survival` package is reported in the [requirements.txt](requirements.txt).

## Installation

Python version supported : ![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-green)

This project requires **Python 3.8, 3.9 or 3.10** (3.9+ recommended due to dependencies such as `scikit-survival` and `pyradiomics`).

Due to the dependencies of `pyradiomics` and `scikit-survival`, installation requires a C++ compiler to be present on the system.

To install `nsclc_survival` package in `Python` run:
```
git clone https://github.com/irene-ballantini/nsclc_survival

pip install --editable .
```

## Usage
Command Line Interface

## How to Cite 

## Author