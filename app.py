import streamlit as str
import os
import pickle
import pandas as pd

# 1. Force absolute tracking based on where app.py lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. App Header
str.title("🏡 House Price Prediction Web App")
str.write("Enter the house details below to estimate its market price instantly.")

# 3. Load our trained model using a bulletproof absolute path
model_path = os.path.join(script_dir, 'models', 'best_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

str.markdown("---")

# 4. User Input Forms
str.subheader("📋 Step 1: Provide House Specifications")

col1, col2 = str.columns(2)

with col1:
    area = str.number_input("📐 Total Area (in sq ft):", min_value=100, max_value=10000, value=1500, step=50)
    bedrooms = str.selectbox("🛏️ Number of Bedrooms:", [1, 2, 3, 4, 5], index=2)
    bathrooms = str.selectbox("🚿 Number of Bathrooms:", [1, 2, 3, 4, 5], index=1)

with col2:
    age = str.number_input("⏳ Age of the Property (in years):", min_value=0, max_value=100, value=5, step=1)
    location = str.selectbox("📍 Location / Area Type:", ["City Center", "Suburb", "Rural"])
    prop_type = str.selectbox("🏠 Property Type:", ["Apartment", "House", "Villa"])

str.markdown("---")

# 5. Prediction Trigger
str.subheader("🎯 Step 2: Generate Evaluation")

# Create a clean row for our button
if str.button("🚀 Calculate Estimated House Price"):
    
    # Put the user inputs into a small DataFrame so the model can read it
    input_data = pd.DataFrame([{
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'Location': location,
        'Property_Type': prop_type
    }])
    
    # Send the raw data into our saved pipeline factory
    predicted_price = model.predict(input_data)[0]
    
    # Display the final valuation beautifully on screen
    str.balloons() # Throws a fun celebration effect on screen!
    str.success(f"📊 Estimated Property Valuation: **₹{predicted_price:,.2f}**")