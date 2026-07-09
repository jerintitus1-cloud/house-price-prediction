# 🏠 Indian Housing Price Prediction System

A production-grade, modular Machine Learning system built with Python to analyze and predict real estate valuations based on key property features.

## 📊 Performance Insights
- **Winning Algorithm:** Random Forest Regressor 
- **Model Accuracy ($R^2$ Score):** 0.974 (Explains over 97% of data variance)
- **Primary Pricing Drivers:** 1. Property Area ($68.76\%$)
  2. City Center Location ($15.52\%$)

## 📁 Repository Structure
- `data/` : Contains the housing dataset source.
- `src/data_preprocessing.py` : Handles raw ingestion and filters identifiers.
- `src/feature_engineering.py` : Manages numerical scaling and categorical One-Hot Encoding pipelines.
- `src/model_training.py` : Evaluates and benchmarks multiple regressor algorithms.
- `src/model_optimization.py` : Analyzes feature importance weights and saves the trained pipeline binary.
- `src/model_inference.py` : Exposes a scalable runtime interface for loading model artifacts.
- `src/web_app.py` : Streamlit UI layout implementing production validation checks.

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip3 install pandas scikit-learn streamlit