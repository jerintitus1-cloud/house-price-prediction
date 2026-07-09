import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle
import sys
import os

# Allow importing from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_clean_data
from feature_engineering import build_preprocessing_pipeline

def finalize_and_save_model():
    # 1. Fetch raw data
    df = load_and_clean_data()
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Get our custom preprocessing rules
    preprocessor = build_preprocessing_pipeline()
    
    # 3. Create a master Pipeline bundle (Preprocessor + Winning Model together)
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    print("\n🏗️ Training the final Random Forest pipeline model...")
    final_pipeline.fit(X_train, y_train)
    
    # 4. Extract Feature Importances from the pipeline
    # We pull feature names directly out of the preprocessor to match them correctly
    feature_names = (
        final_pipeline.named_steps['preprocessor']
        .transformers_[0][2] +  # Numeric column names
        list(final_pipeline.named_steps['preprocessor']
             .transformers_[1][1]
             .named_steps['onehot']
             .get_feature_names_out(['Location', 'Property_Type'])) # Categorical names
    )
    
    importances = final_pipeline.named_steps['regressor'].feature_importances_
    
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    print("=" * 45)
    sorted_indices = np.argsort(importances)[::-1]
    for i in sorted_indices:
        print(f"{feature_names[i]:<25}: {importances[i]*100:.2f}% importance")
    print("=" * 45)
    
    # 5. Model Persistence: Save the complete pipeline into the 'models' folder
    model_save_path = '/Users/apple/Desktop/python-test/intern project/4th month/models/model.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump(final_pipeline, f)
        
    print(f"💾 Success! Complete model pipeline saved to:\n   {model_save_path}")

if __name__ == "__main__":
    finalize_and_save_model()