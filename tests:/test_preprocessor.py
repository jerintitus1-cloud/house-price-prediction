import os
import pandas as pd
from src.data_preprocessor import load_and_split_data, build_preprocessor_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'house_prices.csv')

def test_data_splitting():
    """Validates that training and testing vectors split exactly as configured."""
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, test_size=0.2)
    
    assert len(X_train) > 0, "Training feature vector shouldn't be empty."
    assert len(X_test) > 0, "Testing feature vector shouldn't be empty."
    assert len(X_train) + len(X_test) == len(pd.read_csv(DATA_PATH)), "Row splits match original dataset total size."

def test_preprocessor_pipeline_structure():
    """Validates that the transformation step maps all intended core parameters."""
    preprocessor = build_preprocessor_pipeline()
    transformers = [t[0] for t in preprocessor.transformers]
    
    assert 'num' in transformers, "Numerical processing step missing."
    assert 'cat' in transformers, "Categorical processing step missing."