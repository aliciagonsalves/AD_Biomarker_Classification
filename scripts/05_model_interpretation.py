# scripts/05_model_interpretation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import json
from sklearn.inspection import permutation_importance

def generate_interpretation_plots():
    """
    Loads the final tuned models and test data to generate feature
    importance and interpretation plots for each model architecture.
    """
    print("--- Running Script 05: Generating Final Model Interpretation Plots ---")

    # --- 1. Configuration & Style ---
    tables_dir = 'results/tables'
    models_dir = 'results/models'
    figures_dir = 'results/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    COLOR_LR = "#1252E8"
    COLOR_XGB = "#EB130F"
    COLOR_SVM = "#6B18E7"
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # --- 2. Load Data and Models ---
    try:
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
        
        print("Loaded test data and all three tuned models successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: A required file is missing. Please run the full pipeline first. Details: {e}")
        return

    # --- 3. Prepare Test Data ---
    test_df = test_df_raw.copy()
    test_df['pTau_Abeta42_ratio'] = test_df['pTau'] / (test_df['Abeta42'] + 1e-6)
    test_df['Abeta42_Abeta40_ratio'] = test_df['Abeta42'] / (test_df['Abeta40'] + 1e-6)
    test_df['pTau_tTau_ratio'] = test_df['pTau'] / (test_df['tTau'] + 1e-6)
    y_test = test_df['AD_Pathology']

    # --- Plot 1: Logistic Regression Coefficients ---
    print("\n[1/3] Generating feature importance plot for Logistic Regression...")
    lr_model = models['Logistic Regression']
    lr_features = features['Logistic Regression']
    lr_coeffs = pd.DataFrame({
        'feature': lr_features,
        'coefficient': lr_model.named_steps['model'].coef_[0]
    }).sort_values('coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=lr_coeffs, color=COLOR_LR)
    plt.title('Feature Importances (Coefficients) for Logistic Regression', fontsize=16, weight='bold')
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_lr_coefficients.png'), dpi=300)
    print("  > Logistic Regression plot saved.")
    plt.close()

    # --- Plot 2: SVM Permutation Importance ---
    print("\n[2/3] Generating feature importance plot for SVM...")
    svc_model = models['SVM']
    svc_features = features['SVM']
    X_test_svc = test_df[svc_features]
    
    # We need to scale the data for permutation importance calculation
    X_test_svc_scaled = svc_model.named_steps['scaler'].transform(X_test_svc)
    
    perm_result = permutation_importance(
        svc_model.named_steps['model'], X_test_svc_scaled, y_test, 
        n_repeats=30, random_state=42, n_jobs=-1, scoring='accuracy'
    )
    
    perm_importance_df = pd.DataFrame({
        'feature': X_test_svc.columns,
        'importance_mean': perm_result.importances_mean,
    }).sort_values(by="importance_mean", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance_mean', y='feature', data=perm_importance_df, color=COLOR_SVM)
    plt.title('Permutation Feature Importance for SVM Model', fontsize=16, weight='bold')
    plt.xlabel('Mean Accuracy Decrease', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05_svm_permutation_importance.png'), dpi=300)
    print("  > SVM plot saved.")
    plt.close()

    # --- Plot 3: XGBoost SHAP Summary Plot ---
    print("\n[3/3] Generating SHAP summary plot for XGBoost model...")
    xgb_model = models['XGBoost']
    xgb_features = features['XGBoost']
    X_test_xgb = test_df[xgb_features]
    
    X_scaled_shap = xgb_model.named_steps['scaler'].transform(X_test_xgb)
    explainer = shap.TreeExplainer(xgb_model.named_steps['model'])
    shap_values = explainer.shap_values(X_scaled_shap)
    
    shap.summary_plot(shap_values, pd.DataFrame(X_scaled_shap, columns=X_test_xgb.columns), show=False, plot_size=[10,6], max_display=len(xgb_features))
    plt.title("SHAP Feature Importance for XGBoost Model", fontsize=16)
    plt.savefig(os.path.join(figures_dir, '06_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    print("  > SHAP summary plot saved.")
    plt.close()

    print("\n--- Script 05 Complete: All interpretation plots generated. ---")

if __name__ == '__main__':
    generate_interpretation_plots()