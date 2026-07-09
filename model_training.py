import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import sys
import os

# Allow importing from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_and_clean_data
from feature_engineering import build_preprocessing_pipeline

def train_and_evaluate_all():
    # 1. Fetch data and build preprocessing rules
    df = load_and_clean_data()
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = build_preprocessing_pipeline()
    
    # 2. Transform the training data and validation data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # 3. Initialize our 3 models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print("\n🎯 STARTING MODEL EVALUATION REPORT:")
    print("=" * 45)
    
    # 4. Train each model, predict, and compute metrics
    for name, model in models.items():
        # Train
        model.fit(X_train_transformed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_transformed)
        
        # Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"🤖 Model: {name}")
        print(f"   • Mean Absolute Error (MAE): ₹{mae:,.2f}")
        print(f"   • Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
        print(f"   • R² Score (Accuracy Metric): {r2:.3f}")
        print("-" * 45)

if __name__ == "__main__":
    train_and_evaluate_all()