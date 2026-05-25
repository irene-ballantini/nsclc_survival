#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_features_to_csv(features_list, output_path):
    """
    Converts a list of dictionaries to a Pandas DataFrame and saves it as a CSV file.
    Args:
        features_list (list of dict): List where each element is a dictionary of features for a single patient.
        output_path (str or Path): Path to save the CSV file.
    """
    output_path = Path(output_path)

    if features_list is None or not features_list:
        print("[WARNING] No features to save.")
        return

    # DataFrame creation
    df = pd.DataFrame(features_list)
    
    # PatientID as first column
    cols = ["PatientID"] + [c for c in df.columns if c != "PatientID"]
    df = df[cols]
    
    # Saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] File saved successfully in: {output_path}")
    print(f"[INFO] Total columns written: {df.shape[1]} (PatientID + {df.shape[1] - 1} features)")

def plot_extreme_survival_curves(survival_functions, risk_scores, output_path):
    """
    Plot the predicted survival functions for the highest and lowest risk patients.

    Args:
        survival_functions (array-like): Array of StepFunction objects 
            (from scikit-survival) representing individual survival curves.
        risk_scores (array-like): Computed risk scores corresponding to the functions.
    """
    if survival_functions is None or len(survival_functions) == 0:
        print("[WARNING] No survival functions to plot. Make sure 'return_survival_curves=True' was used.")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Identify the indices of the patients with the absolute highest and lowest risk
    idx_max = np.argmax(risk_scores)
    idx_min = np.argmin(risk_scores)

    # Extract the specific curves using the identified indices
    high_risk_curve = survival_functions[idx_max]
    low_risk_curve = survival_functions[idx_min]

    # Plot the curves
    plt.figure(figsize=(8, 5))
    plt.step(high_risk_curve.x, high_risk_curve.y, where="post", 
             label=f"High Risk (Score: {risk_scores[idx_max]:.2f})", color="red")
    plt.step(low_risk_curve.x, low_risk_curve.y, where="post", 
             label=f"Low Risk (Score: {risk_scores[idx_min]:.2f})", color="blue")
    
    plt.title("Individual Survival Curves - Risk Comparison (highest vs lowest risk)") 
    plt.xlabel("Time (Days)")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(output_path)
    #plt.show()