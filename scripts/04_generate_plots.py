# scripts/04_generate_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import roc_curve, auc
import os
import json
import warnings

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


def generate_final_plots():
    """
    Loads all final data and models to generate figures for the
    final report, including exploratory plots and ROC curves on the test set.
    """
    print("--- Running Script 04: Generating Final Report Figures ---")

    # --- 1. Configuration & Global Style ---
    tables_dir = 'results/tables'
    models_dir = 'results/models'
    figures_dir = 'results/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Using your final preferred color scheme for consistency
    COLOR_LR = "#1252E8"      
    COLOR_XGB = "#EB130F"     
    COLOR_SVM = "#6B18E7"      
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

    # --- 2. Load All Necessary Files ---
    try:
        print("Loading all necessary data files...")
        df_cleaned = pd.read_csv(os.path.join(tables_dir, 'cleaned_biomarker_data.csv'))
        test_df_raw = pd.read_csv(os.path.join(tables_dir, 'test_dataset.csv'))
        
        models_to_load = {"Logistic Regression": "logistic_regression", "XGBoost": "xgboost", "SVM": "svm"}
        models = {}
        features = {}

        for display_name, key_name in models_to_load.items():
            model_path = os.path.join(models_dir, f'{key_name}_best.joblib')
            features_path = os.path.join(models_dir, f'{key_name}_best_features.json')
            models[display_name] = joblib.load(model_path)
            with open(features_path, 'r') as f:
                features[display_name] = json.load(f)
        
        print("  > All data and models loaded successfully.")
    except FileNotFoundError as e:
        print(f"  > ERROR: A required file is missing. Please run the full pipeline first (01-02). Details: {e}")
        return
        
    # --- 3. Generate Exploratory Data Analysis (EDA) Plots ---
    print("\n[Step 1/2] Generating EDA plots...")
    
    # Distribution Plots using the full cleaned dataset
    for col in ['Abeta40', 'Abeta42', 'tTau', 'pTau']:
        plt.figure(figsize=(6, 4))
        sns.histplot(df_cleaned[col].dropna(), kde=True, bins=20, color=COLOR_LR)
        plt.title(f'Distribution of {col} (Cleaned Data)', fontsize=14)
        plt.savefig(os.path.join(figures_dir, f'01_eda_distribution_{col}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print("  > Biomarker distribution plots saved.")
        
    # Correlation Heatmap using the full cleaned dataset
    corr_features = ['Abeta40', 'Abeta42', 'tTau', 'pTau', 'Age at Death']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cleaned[corr_features].corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Biomarkers and Age', fontsize=16, weight='bold')
    plt.savefig(os.path.join(figures_dir, '02_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    print("  > Correlation heatmap saved.")
    plt.close()

    # --- 4. Prepare Test Data for Model Evaluation ---
    # Engineer features on the test set
    test_df = test_df_raw.copy()
    test_df['pTau_Abeta42_ratio'] = test_df['pTau'] / (test_df['Abeta42'] + 1e-6)
    test_df['Abeta42_Abeta40_ratio'] = test_df['Abeta42'] / (test_df['Abeta40'] + 1e-6)
    test_df['pTau_tTau_ratio'] = test_df['pTau'] / (test_df['tTau'] + 1e-6)
    y_test = test_df['AD_Pathology']

    # --- 5. Generate Model Performance Plot (ROC Curves) on Test Data ---
    print("\n[Step 2/2] Generating final model performance plot (ROC Curves on test data)...")
    plt.figure(figsize=(9, 9))
    
    model_colors = {"Logistic Regression": COLOR_LR, "XGBoost": COLOR_XGB, "SVM": COLOR_SVM}

    for name, model in models.items():
        # Select the specific features this model was trained on
        model_features = features[name]
        X_test = test_df[model_features]
        
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=model_colors[name], lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Final Model Performance on Held-Out Test Set', fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=11); plt.grid(alpha=0.5)
    plt.savefig(os.path.join(figures_dir, '03_roc_curve_comparison.png'), dpi=300)
    print("  > ROC curve comparison plot saved.")
    plt.close()
    
    print("\n--- Script 04 Complete: All figures generated. ---")

if __name__ == '__main__':
    generate_final_plots()