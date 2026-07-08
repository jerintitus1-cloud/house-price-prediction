# 📖 House Price Prediction Data Dictionary

This document defines the features and target variables included in the `house_prices.csv` dataset used for training the house pricing machine learning model.

---

### Features (Model Inputs)

| Column Name | Data Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| **Area** | Integer | The total layout size of the property calculated in square feet (sq ft). | `100` to `10000` |
| **Bedrooms** | Integer | Total count of dedicated bedrooms available in the property. | `1`, `2`, `3`, `4`, `5` |
| **Bathrooms** | Integer | Total count of functional bathrooms constructed in the house. | `1`, `2`, `3`, `4`, `5` |
| **Age** | Integer | The absolute historical age of the building since construction (in years). | `0` (Brand New) to `100` |
| **Location** | Categorical | The classified neighborhood setting where the building is located. | `City Center`, `Suburb`, `Rural` |
| **Property_Type** | Categorical | The structural design classification of the residential building. | `Apartment`, `House`, `Villa` |

---

### Target Variable (Model Output)

| Column Name | Data Type | Description | Target Unit |
| :--- | :--- | :--- | :--- |
| **Price** | Float / Int | The final historical or evaluated market valuation of the property. | Indian Rupees (₹) |