# scripts/02_create_datasets.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

print("--- Running Script 02: Creating Train/Test Datasets ---")
df = pd.read_csv('results/tables/cleaned_biomarker_data.csv')

# Binarize the target for stratification
target_col = 'Overall AD neuropathological Change'
df_ml = df.dropna(subset=[target_col]).copy()
df_ml['AD_Pathology'] = df_ml[target_col].apply(lambda x: 1 if x == 'High' else 0)

# Create a stratified split
train_df, test_df = train_test_split(
    df_ml, 
    test_size=0.25, 
    random_state=42, 
    stratify=df_ml['AD_Pathology']
)

# Save the datasets
train_df.to_csv('results/tables/train_dataset.csv', index=False)
test_df.to_csv('results/tables/test_dataset.csv', index=False)

print(f"Data split complete. Training set has {len(train_df)} samples.")
print(f"Testing set has {len(test_df)} samples.")
print("Saved 'train_dataset.csv' and 'test_dataset.csv'.")
print("\n--- Script 02 Complete ---")