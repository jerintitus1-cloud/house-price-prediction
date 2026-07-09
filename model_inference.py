import pickle
import pandas as pd
import os

def predict_house_price(input_data):
    """
    Loads the trained pickle pipeline and predicts the house price.
    input_data: A dictionary containing the house features.
    """
    model_path = '/Users/apple/Desktop/python-test/intern project/4th month/models/model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find serialized model at {model_path}")
        
    # 1. Load the saved master pipeline
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    # 2. Convert incoming input dictionary into a pandas DataFrame format matching training
    df_input = pd.DataFrame([input_data])
    
    # 3. Predict using the pipeline (it will automatically scale & one-hot encode behind the scenes!)
    prediction = pipeline.predict(df_input)[0]
    
    return prediction

if __name__ == "__main__":
    # Let's test a mock sample house input to verify our inference pipeline works!
    sample_house = {
        'Area': 2500,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age': 10,
        'Location': 'City Center',
        'Property_Type': 'House'
    }
    
    try:
        estimated_price = predict_house_price(sample_house)
        print("\n🚀 INFERENCE ENGINE TEST SUCCESSFUL!")
        print("=" * 45)
        print(f"🏠 Input House Details: 2500 sqft, 3 BHK in City Center")
        print(f"💰 Model Predicted Price: ₹{estimated_price:,.2f}")
        print("=" * 45)
    except Exception as e:
        print(f"❌ Inference engine failed with error: {e}")