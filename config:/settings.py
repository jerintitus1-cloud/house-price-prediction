"""
Centralized Configuration Settings for House Price Prediction Pipeline.
Handles system directories, feature boundaries, and interface parameters.
"""
import os

# --- Core System Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

DATA_FILE_PATH = os.path.join(DATA_DIR, 'house_prices.csv')
MODEL_FILE_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')

# --- Feature Engineering Groups ---
TARGET_COLUMN = 'Price'
NUMERICAL_FEATURES = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
CATEGORICAL_FEATURES = ['Location', 'Property_Type']

# --- UI Interface Configurations ---
APP_TITLE = "🏡 House Price Prediction Web App"
APP_SUBTITLE = "Enter the house details below to estimate its market price instantly."

LOCATION_OPTIONS = ["City Center", "Suburb", "Rural"]
PROPERTY_OPTIONS = ["Apartment", "House", "Villa"]

# --- Hyperparameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100