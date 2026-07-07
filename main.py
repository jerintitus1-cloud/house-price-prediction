import pandas as pd
import os
import shutil
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

print("--- Step 1: Structuring Data Science Directories ---")
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define our directory blueprint
old_csv_path = os.path.join(script_dir, 'house_prices.csv')
data_folder_path = os.path.join(script_dir, 'data')
notebooks_folder_path = os.path.join(script_dir, 'notebooks')
new_csv_path = os.path.join(data_folder_path, 'house_prices.csv')

# 1. Setup 'notebooks/' directory for EDA scratchpads
if not os.path.exists(notebooks_folder_path):
    os.makedirs(notebooks_folder_path, exist_ok=True)
    print("📁 Created 'notebooks/' folder for your exploratory Jupyter Notebooks.")

# 2. Setup 'data/' directory and route dataset
if os.path.exists(old_csv_path):
    os.makedirs(data_folder_path, exist_ok=True)
    shutil.move(old_csv_path, new_csv_path)
    print("🚚 Moved 'house_prices.csv' into its clean 'data/' home.")

# 3. Load dataset from the finalized repository structural path
if os.path.exists(new_csv_path):
    df = pd.read_csv(new_csv_path)
    print("✅ Dataset verified and loaded from 'data/house_prices.csv'.\n")
    
    # Preprocessing & Splitting Columns
    X = df.drop(['Property_ID', 'Price'], axis=1)
    y = df['Price']
    
    numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    categorical_features = ['Location', 'Property_Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Connect Engine & Train Pipeline
    print("--- Step 2: Training Model Engine ---")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("⏳ Training the final model pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    # Save the pipeline to disk
    print("\n--- Step 3: Saving Model to Hard Drive ---")
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'best_model.pkl')
    with open(model_save_path, 'wb') as file:
        pickle.dump(model_pipeline, file)
        
    print(f"💾 Success! Trained model pipeline is safely saved.")
    print("🎉 File hierarchy completely structured!")
    
else:
    print(f"✅ Folders verified. Dataset already correctly resting inside 'data/' folder.")