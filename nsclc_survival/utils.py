#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

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

def plot_deviance_residuals(df_risk_residuals, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Se nel tuo dataframe il Risk Score non è presente, lo recuperiamo calcolandolo 
    # o passandolo. Assumiamo che tu lo abbia unito o che sia presente. 
    # Se hai usato il df dei residui puro, ricordati che contiene 'Cumulative_Hazard_Predicted'.
    # Per consistenza, usiamo il logaritmo del rischio cumulativo o, se lo hai salvato, 
    # il 'Risk_Score' (Rad-Score) dal df_risk_scores.
  
    plt.figure(figsize=(10, 6))
    
    # Se passi il df dei residui puro, possiamo usare il log del Cumulative Hazard 
    # che è matematicamente proporzionale al Risk Score lineare del modello di Cox.
    # Se hai già una colonna 'Risk_Score', usa quella!
    if 'Risk_Score' in df_risk_residuals.columns:
        x_values = df_risk_residuals['Risk_Score']
        x_label = 'Risk Score (Rad-Score)'
    else:
        # Failsafe: usiamo il log del rischio cumulativo predetto
        x_values = np.log(df_risk_residuals['Cumulative_Hazard_Predicted'])
        x_label = 'Log(Predicted Cumulative Hazard)'

    y_values = df_risk_residuals['Deviance_Residual']
    status = df_risk_residuals['Event_Status'].values

    is_event = (status == True)
    is_censored = (status == False)
  
    #Definiamo i colori in base allo stato dell'evento (0 = Censurato, 1 = Evento)
    #Rende il grafico molto più informativo dal punto di vista clinico!  
    status_labels = df_risk_residuals['Event_Status'].map({
        True: 'Event (Deceased)', 
        False: 'Censored (Alive)'
    })

    # 3. Disegno dei punti dei Censurati (BLU)
    plt.scatter(
        x=x_values[is_censored],
        y=y_values[is_censored],
        color='#1f77b4',
        alpha=0.8,
        edgecolors='w',
        s=70,
        label='Censored (Alive)' # L'etichetta è legata direttamente al colore!
    )

    # 4. Disegno dei punti degli Eventi (ROSSO)
    plt.scatter(
        x=x_values[is_event],
        y=y_values[is_event],
        color='#d62728',
        alpha=0.8,
        edgecolors='w',
        s=70,
        label='Event (Deceased)' # L'etichetta è legata direttamente al colore!
    )
   
    # Linea di riferimento sullo zero (modello ideale)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
    
    # Linee di soglia critica per gli outlier statistici (+2 e -2)
    plt.axhline(y=2, color='gray', linestyle='--', linewidth=1, label='Outlier Threshold ($\pm$2)')
    plt.axhline(y=-2, color='gray', linestyle='--')
    
    # Formattazione grafica
    plt.title('Model Diagnostics: Risk Score vs Deviance Residuals', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Deviance Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Sistemazione della legenda
    plt.legend(title='Patient Status', loc='best')
    
    # Ottimizzazione spazi e salvataggio
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"[INFO] Residuals diagnostic plot saved to: {output_path}")

def kaplan_meier_plot(y_test, pred_classes, logrank_p_value, output_path, title_suffix="DeepCox"):
    """_summary_

    Args:
        y_test (_type_): _description_
        pred_classes (_type_): _description_
        logrank_p_value (_type_): _description_
        output_path (_type_): _description_
        title_suffix (str, optional): _description_. Defaults to "DeepCox".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    times_test = y_test['Survival_Time']
    events_test = y_test['Event_Status'].astype(int)
    
    idx_low = (pred_classes == 0)
    idx_high = (pred_classes == 1)
    
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()
    
    plt.figure(figsize=(8, 5))
    
    # Low Risk Curve
    kmf_low.fit(times_test[idx_low], event_observed=events_test[idx_low], 
                label=f'Low Risk (n={np.sum(idx_low)})')
    kmf_low.plot_survival_function(ci_show=True, color='tab:blue')
    
    # High Risk Curve
    kmf_high.fit(times_test[idx_high], event_observed=events_test[idx_high], 
                 label=f'High Risk (n={np.sum(idx_high)})')
    kmf_high.plot_survival_function(ci_show=True, color='tab:red')
    
    plt.title(f'Stratification of the Population ({title_suffix})\nLog-Rank p-value: {logrank_p_value:.5f}')
    plt.xlabel('Survival Time (Days)')
    plt.ylabel('Survival Probability')
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    
    plt.savefig(output_path)
    #plt.show()

def plot_loss_funct(loss_history, output_path):
    """
    Plot the trend of the loss function

    Args:
        loss_history (list): list containing loss function values across the epochs
        output_path (Path): Saving path
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='#1f77b4', linewidth=2, label='Training Loss')
    plt.title('Deep Cox Model - Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Negative Log-Likelihood Loss', fontsize=12) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    plt.savefig(output_path)