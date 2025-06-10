# 💼 Sales Revenue Prediction

This project focuses on predicting **Total Revenue** from US regional sales data using machine learning techniques. The dataset includes sales information such as order quantity, discount, unit price, dates, warehouse codes, and sales channels.

I engineered insightful features, handled high-cardinality IDs with target encoding, and evaluated models using robust cross-validation. The final model is deployed as an interactive **Streamlit web app** for real-time prediction.

---

## Problem Statement

Businesses often face challenges in forecasting revenue due to complex relationships between pricing, discounts, customer behavior, and logistics. Our goal was to build a predictive model that helps estimate revenue from sales order data efficiently and accurately.

---

## Features & Engineering

- Parsed and cleaned currency and date columns
- Engineered time-based features like `Order_to_Delivery_Days`
- Computed `Total Revenue` as the prediction target
- Target encoded high-cardinality identifiers like Product ID and Customer ID
- One-hot encoded categorical features like Sales Channel and Warehouse

---

## Models Evaluated

### 🔸 Random Forest Regressor  
- Train R² : 0.9998  
- CV R²    : 0.9979  
- Test R²  : 0.9984  
- MAE      : 151.72  
- RMSE     : 351.06
  
### 🔹 Linear Regression  
- Train R² : 0.8477  
- CV R²    : 0.8465  
- Test R²  : 0.8442  
- MAE      : 2512.11  
- RMSE     : 3512.04



The Random Forest model significantly outperformed Linear Regression and was chosen as the final model.

---

## Deployment

The final model was deployed using **Streamlit** with the following functionalities:
- Interactive UI for sales input
- Real-time prediction of revenue
- Backend integration using `pickle`-saved model and encodings

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
