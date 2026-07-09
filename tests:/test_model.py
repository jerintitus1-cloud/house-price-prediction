import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Look dynamically for local or root layout configurations
local_path = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
cloud_path = os.path.join(BASE_DIR, 'best_model.pkl')
MODEL_PATH = local_path if os.path.exists(local_path) else cloud_path

def test_model_loading_and_prediction():
    """Confirms that the pickel file loads successfully and processes raw input data formats."""
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        
    assert model is not None, "Trained model file failed to deserialize or load."
    
    # Simulate a single mock user input row precisely resembling app interface specifications
    mock_user_input = pd.DataFrame([{
        'Area': 1500,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age': 5,
        'Location': 'City Center',
        'Property_Type': 'Apartment'
    }])
    
    prediction = model.predict(mock_user_input)
    
    assert len(prediction) == 1, "Model output array dimension mismatch."
    assert prediction[0] > 0, "The calculated house prediction value should be a positive number."