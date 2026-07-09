import streamlit as st
import sys
import os

# Allow importing directly from the src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_inference import predict_house_price

# 1. UI Page Configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 Indian Housing Price Prediction System")
st.markdown("Enter the exact property specifications below to generate a production-ready machine learning valuation estimate.")
st.markdown("---")

# 2. Input Validation Form
with st.form("prediction_form"):
    st.subheader("📋 Property Dimensions & Location Features")
    
    # Numerical features with bounds validation
    area = st.number_input("Total Area (in Square Feet):", min_value=100, max_value=10000, value=2500, step=50)
    bedrooms = st.slider("Number of Bedrooms (BHK):", min_value=1, max_value=5, value=3)
    bathrooms = st.slider("Number of Bathrooms:", min_value=1, max_value=3, value=2)
    age = st.number_input("Property Age (in Years):", min_value=0, max_value=100, value=5, step=1)
    
    # Categorical option selections matching your dataset columns
    location = st.selectbox("Geographic Location Category:", ["City Center", "Suburb", "Rural"])
    property_type = st.selectbox("Structural Property Type:", ["House", "Villa", "Apartment"])
    
    # Submit handler
    submit_button = st.form_submit_button(label="🔮 Generate Value Estimate")

# 3. Form Submission Handling with Error Protections
if submit_button:
    # Package input dictionary
    user_inputs = {
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'Location': location,
        'Property_Type': property_type
    }
    
    with st.spinner("Processing features through ML Pipeline..."):
        try:
            # Route inputs to background inference script
            predicted_valuation = predict_house_price(user_inputs)
            
            # Display success and results cleanly
            st.success("✅ Target Property Valuation Completed Successfully!")
            st.metric(
                label=f"Predicted Marketplace Value ({location} {property_type})", 
                value=f"₹{predicted_valuation:,.2f}"
            )
            
        except Exception as error:
            st.error(f"❌ Production Error encountered during calculation: {error}")