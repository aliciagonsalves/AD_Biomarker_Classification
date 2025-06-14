# scripts/01_data_processing.py

import pandas as pd
import os

def process_raw_data():
    """
    Loads raw Luminex biomarker and donor metadata Excel files, merges them,
    cleans the data by handling outliers based on the 1.5*IQR rule, and saves
    a final, processed CSV file ready for modeling.
    """
    print("--- Running Script 01: Data Loading and Cleaning ---")

    # --- 1. Configuration & Setup ---
    data_dir = 'data'
    luminex_file = os.path.join(data_dir, 'sea-ad_cohort_mtg-tissue_extractions-luminex_data.xlsx')
    metadata_file = os.path.join(data_dir, 'sea-ad_cohort_donor_metadata_072524.xlsx')

    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'cleaned_biomarker_data.csv')
    print(f"Output will be saved to: {output_file}")

    # --- 2. Load and Prepare Luminex Data ---
    print("\n[Step 1/5] Loading and preparing Luminex data...")
    try:
        # Header is on the second row of the Excel file (index 1)
        raw_luminex = pd.read_excel(luminex_file, header=1)
        # Select only the first 5 columns (Donor ID + RIPA buffer biomarkers)
        luminex_df = raw_luminex.iloc[:, :5].copy()
        luminex_df.columns = ['Donor ID', 'Abeta40', 'Abeta42', 'tTau', 'pTau']
        # Convert biomarker columns to numeric, coercing any errors
        for col in ['Abeta40', 'Abeta42', 'tTau', 'pTau']:
            luminex_df[col] = pd.to_numeric(luminex_df[col], errors='coerce')
        print("  > Luminex data prepared successfully.")
    except FileNotFoundError:
        print(f"  > ERROR: Luminex file not found at '{luminex_file}'. Please check path and filename.")
        return

    # --- 3. Load and Prepare Metadata ---
    print("\n[Step 2/5] Loading and preparing donor metadata...")
    try:
        df_metadata = pd.read_excel(metadata_file)
        # Select only the columns needed for merging and modeling
        meta_cols_to_select = ['Donor ID', 'Age at Death', 'Overall AD neuropathological Change']
        metadata_df = df_metadata[meta_cols_to_select].copy()
        print("  > Metadata prepared successfully.")
    except FileNotFoundError:
        print(f"  > ERROR: Metadata file not found at '{metadata_file}'. Please check path and filename.")
        return
    except KeyError as e:
        print(f"  > ERROR: A required column was not found in the metadata file. Details: {e}")
        return

    # --- 4. Merge Datasets ---
    print("\n[Step 3/5] Merging Luminex and metadata...")
    luminex_df['Donor ID'] = luminex_df['Donor ID'].astype(str).str.strip()
    metadata_df['Donor ID'] = metadata_df['Donor ID'].astype(str).str.strip()
    merged_df = pd.merge(luminex_df, metadata_df, on='Donor ID', how='left')
    print(f"  > Initial merged data contains {len(merged_df)} rows.")

    # --- 5. Outlier Removal ---
    print("\n[Step 4/5] Removing outliers using the 1.5*IQR rule...")
    df_to_clean = merged_df.copy()
    initial_rows = len(df_to_clean)
    
    for col in ['Abeta40', 'Abeta42', 'tTau', 'pTau']:
        # Drop rows with NaN in the current column before calculating quantiles
        col_data = df_to_clean[col].dropna()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df_to_clean[col] < lower_bound) | (df_to_clean[col] > upper_bound)
            df_to_clean = df_to_clean[~outlier_mask]

    final_rows = len(df_to_clean)
    print(f"  > Removed {initial_rows - final_rows} rows containing outliers.")
    print(f"  > Final cleaned dataset contains {final_rows} rows.")

    # --- 6. Save Final Cleaned Data ---
    print(f"\n[Step 5/5] Saving cleaned data...")
    cleaned_df = df_to_clean
    cleaned_df.to_csv(output_file, index=False)
    print(f"  > Successfully saved to {output_file}")
    
    print("\n--- Script 01 Complete ---")

if __name__ == '__main__':
    process_raw_data()