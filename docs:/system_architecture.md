# 🏗️ System Architecture Design Guide

This document describes the software design and data pipeline routing mechanics for the House Price Prediction system.

## 🔄 Core Engineering Data Flow

1. **Ingestion Layer (`data/`)**: Raw housing parameters are collected via `house_prices.csv`.
2. **Pipeline Processing & Engineering (`src/data_preprocessor.py`)**: 
   * Continual values (`Area`, `Age`) are scaled using `StandardScaler`.
   * String text items (`Location`, `Property_Type`) are translated via `OneHotEncoder`.
3. **Training Factory (`src/model_trainer.py`)**: Wraps layout operations into an integrated `Pipeline` alongside a `RandomForestRegressor`.
4. **Binary Serialized Storage (`models/`)**: Outputs the finalized production machine learning engine to `best_model.pkl`.
5. **Frontend Presentation (`app.py`)**: Pulls the model binary file to calculate live user requests on the Streamlit screen interface.