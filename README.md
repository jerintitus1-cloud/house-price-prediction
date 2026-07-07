# 🏡 House Price Prediction System

An end-to-end Machine Learning web application built using Python to estimate residential real estate market valuations based on property specifications.

## 📊 Performance Summary
* **Algorithm Used:** Random Forest Regressor
* **Accuracy Grade ($R^2$ Score):** 97.4%
* **Mean Absolute Error (MAE):** ~₹1,436,717

## 🔍 Core Insights (Feature Importance)
The model determines pricing heavily based on two main criteria:
1. **Total Area (68.76%)** - The primary driving force behind the valuation.
2. **City Center Location (15.52%)** - Premium urban placement significantly drives up costs.

---

## 🛠️ Project Structure
* `data/` - Contains raw and processed datasets (e.g., `house_prices.csv`).
* `notebooks/` - Dedicated directory for Jupyter notebooks and exploratory data analysis (EDA).
* `models/` - Contains the serialized pipeline (`best_model.pkl`) saving our trained configurations.
* `main.py` - The production pipeline script for data loading, splitting, training, and evaluation.
* `app.py` - The interactive Streamlit web interface code.

---

## 🚀 How to Run the Web Application

1. Open your terminal and navigate to this folder.
2. Fire up the application server using the following command:
   ```bash
   python3 -m streamlit run app.py

   python3 -m streamlit run app.py