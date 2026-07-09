import pandas as pd
import os

def load_and_clean_data():
    """
    Loads your house prices dataset using its exact path and name,
    removes identifier columns, and returns a clean DataFrame.
    """
    # The exact path and filename discovered by your Mac terminal
    folder_path = '/Users/apple/Desktop/python-test/intern project/4th month/'
    file_name = 'house_prices (1).csv'
    full_path = os.path.join(folder_path, file_name)
    
    # 1. Load data
    df = pd.read_csv(full_path)
    print(f"📦 Successfully loaded {len(df)} records from {file_name}")
    
    # 2. Drop unique identifier column (Property_ID won't help the AI find pricing patterns)
    if 'Property_ID' in df.columns:
        df = df.drop(columns=['Property_ID'])
        print("✂️ Dropped 'Property_ID' column for training.")
        
    return df

if __name__ == "__main__":
    # Test our new module to make sure it works perfectly on its own
    cleaned_df = load_and_clean_data()
    print("✅ Week 1 Pipeline Complete! Cleaned columns:", cleaned_df.columns.tolist())