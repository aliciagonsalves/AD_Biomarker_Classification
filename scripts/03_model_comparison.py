# scripts/03_model_comparison.py

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
import warnings
import json
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

# --- Methodological Note on Feature Selection ---
# This script uses a rigorous approach to feature engineering and selection.
# An earlier iteration explored L1 (Lasso) regularization. This final version
# instead implements a formal comparison between a feature set derived from
# domain knowledge and another discovered automatically via RFECV, providing
# a more comprehensive evaluation.

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning)

def modeling_pipeline():
    """
    Performs a definitive, multi-stage modeling pipeline on the training set that systematically
    compares feature sets, evaluates three distinct model architectures over
    multiple random seeds, and saves the best-performing version of each model type.
    """
    print("--- Running Script 03: Comprehensive Model & Feature Evaluation ---")

    # --- 1. Configuration & Data Loading ---
    tables_dir = 'results/tables'
    models_dir = 'results/models'
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(os.path.join(tables_dir, 'cleaned_biomarker_data.csv'))
    except FileNotFoundError:
        print(f"ERROR: Cleaned data file not found. Please run Script 01 first.")
        return

    train_file = os.path.join(tables_dir, 'train_dataset.csv')
    try:
        df = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"ERROR: Training data file not found at '{train_file}'. Please run Script 02 first.")
        return

    # --- 2. Feature Engineering & Selection ---
    # Create a full pool of candidate features, including engineered ratios
    df['pTau_Abeta42_ratio'] = df['pTau'] / (df['Abeta42'] + 1e-6)
    df['Abeta42_Abeta40_ratio'] = df['Abeta42'] / (df['Abeta40'] + 1e-6)
    df['pTau_tTau_ratio'] = df['pTau'] / (df['tTau'] + 1e-6)
    full_feature_pool = ['Abeta40', 'Abeta42', 'tTau', 'pTau', 'pTau_Abeta42_ratio', 'Abeta42_Abeta40_ratio', 'pTau_tTau_ratio']
    
    target = 'Overall AD neuropathological Change'
    df_ml = df.dropna(subset=full_feature_pool + [target]).copy()
    df_ml['AD_Pathology'] = np.where(df_ml[target] == 'High', 1, 0)
    X_full, y = df_ml[full_feature_pool], df_ml['AD_Pathology']

    print("\n[Step 1/3] Performing RFECV to find optimal feature subset...")
    estimator_for_rfe = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
    rfe_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    selector = RFECV(estimator_for_rfe, step=1, cv=rfe_cv, scoring='accuracy', n_jobs=-1)
    X_scaled_full = StandardScaler().fit_transform(X_full)
    selector = selector.fit(X_scaled_full, y)
    rfecv_selected_features = list(X_full.columns[selector.support_])
    print(f"  > RFECV selected {selector.n_features_} optimal features: {rfecv_selected_features}")

    # Define the two feature selection strategies we will compare
    feature_sets = {
        "Domain_Knowledge_Set": ['Abeta40', 'Abeta42', 'tTau', 'pTau', 'pTau_Abeta42_ratio'],
        "RFECV_Selected_Set": rfecv_selected_features
    }

    # --- 3. Main Experiment Loop ---
    print("\n[Step 2/3] Tuning & evaluating models across feature sets and random seeds...")
    random_seeds = [0, 42, 123, 1024, 2025]
    all_experiment_results = []
    best_model_registry = {}

    for f_name, f_list in feature_sets.items():
        for seed in random_seeds:
            print(f"  - Running: Features='{f_name}', Seed={seed}")
            X_current = X_full[f_list]
            cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            
            # Define models and parameter grids for tuning
            pipe_lr = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(random_state=seed, class_weight='balanced', max_iter=2000))])
            params_lr = {'model__C': [0.1, 1, 10]}
            scale_pos_weight = y.value_counts().get(0, 1) / y.value_counts().get(1, 1)
            pipe_xgb = Pipeline([('scaler', StandardScaler()), ('model', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, random_state=seed))])
            params_xgb = {'model__max_depth': [2, 3], 'model__n_estimators': [50, 100]}
            pipe_svc = Pipeline([('scaler', StandardScaler()), ('model', SVC(probability=True, random_state=seed, class_weight='balanced'))])
            params_svc = {'model__C': [0.1, 1, 10], 'model__gamma': ['scale']}
            models_to_tune = {"Logistic Regression": (pipe_lr, params_lr), "XGBoost": (pipe_xgb, params_xgb), "SVM": (pipe_svc, params_svc)}
            
            for model_name, (pipe, params) in models_to_tune.items():
                grid_search = GridSearchCV(pipe, params, cv=cv_strat, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_current, y)
                all_experiment_results.append({
                    "Feature_Set": f_name, "Random_Seed": seed, "Model": model_name,
                    "Best_CV_Accuracy": grid_search.best_score_, "Best_Params": str(grid_search.best_params_)
                })
                # Check and save the best version of this model type found so far
                if model_name not in best_model_registry or grid_search.best_score_ > best_model_registry[model_name]['score']:
                    best_model_registry[model_name] = {
                        'score': grid_search.best_score_,
                        'estimator': grid_search.best_estimator_,
                        'features': f_list
                    }

    # --- 4. Final Reporting & Saving ---
    print("\n[Step 3/3] Aggregating results and saving final models...")
    df_results_log = pd.DataFrame(all_experiment_results)
    df_results_log.to_csv(os.path.join(tables_dir, 'full_experiment_log.csv'), index=False)
    print("  > Full experiment log saved.")

    summary_table = df_results_log.groupby(['Feature_Set', 'Model'])['Best_CV_Accuracy'].agg(['mean', 'std']).round(4)
    summary_table = summary_table.sort_values(by='mean', ascending=False)
    print("\n--- FINAL ROBUST PERFORMANCE MATRIX (Averaged over 5 random seeds) ---")
    print(summary_table)
    summary_table.to_csv(os.path.join(tables_dir, 'final_model_performance_summary.csv'))
    
    print("\nSaving best-performing version of each model type...")
    for model_name, details in best_model_registry.items():
        filename_model = f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_best.joblib'
        filename_features = f'{model_name.lower().replace(" ", "_").replace("(", "")}_best_features.json'
        joblib.dump(details['estimator'], os.path.join(models_dir, filename_model))
        with open(os.path.join(models_dir, filename_features), 'w') as f:
            json.dump(details['features'], f)
        print(f"  > Saved best {model_name} (Accuracy: {details['score']:.4f}) and its feature list.")
    
    print("\n--- Script 03 Complete ---")

if __name__ == '__main__':
    modeling_pipeline()