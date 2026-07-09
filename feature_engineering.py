import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
import os

# Allow importing from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_clean_data

def build_preprocessing_pipeline():
    """
    Creates a production-ready transformer that automatically scales numeric features
    and one-hot encodes categorical text features.
    """
    # 1. Identify feature types based on your clean data columns
    numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    categorical_features = ['Location', 'Property_Type']
    
    # 2. Define numerical transformation (Scaling numbers so big values don't break the AI)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 3. Define categorical transformation (Converting text like 'Villa' to 0s and 1s)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 4. Combine them into a single, master preprocessor bundle
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # Let's test loading the raw data and preparing the features
    df = load_and_clean_data()
    
    # Split features (X) and target price (y)
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Split into train/test sets (80% training data, 20% testing data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build our preprocessing rule list
    preprocessor = build_preprocessing_pipeline()
    
    # Try fitting it onto our training features
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    print(f"\n✅ Week 2 Pipeline Ready!")
    print(f"Original training features shape: {X_train.shape}")
    print(f"One-hot encoded training features shape: {X_train_transformed.shape}")