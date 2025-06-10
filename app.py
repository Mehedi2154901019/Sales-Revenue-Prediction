import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# --- Load Trained Artifacts ---
with open('rf_final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Target_Encoding_Maps.pkl', 'rb') as f:
    encoding_maps = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    expected_features = pickle.load(f)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Sales Revenue Predictor", layout="centered")
st.title("💼 Sales Revenue Prediction App")
st.markdown("Enter all required order information to predict total Revenue")

# --- Input Fields ---
sales_team_id = st.selectbox("Sales Team ID", sorted([6,14,21,28,22,12,10,4,23,8,9,5,25,2,7,24,18,20,13,19,17,26,11,15,16,27,3,1]), index=None, placeholder="Select Sales Team ID")
customer_id = st.selectbox("Customer ID", sorted([15,20,16,48,49,21,14,9,33,36,17,32,11,10,30,5,23,46,40,19,22,29,35,42,2,28,34,26,24,18,3,13,4,25,8,47,6,38,1,7,27,44,12,50,43,37,41,31,45,39]), index=None, placeholder="Select Customer ID")
store_id = st.selectbox("Store ID", sorted([259,196,213,107,111,285,6,280,299,261,17,152,317,291,138,354,320,21,349,134,193,282,20,218,173,110,229,238,97,103,305]), index=None, placeholder="Select Store ID")
product_id = st.selectbox("Product ID", sorted([12,27,16,23,26,1,5,46,47,13,38,40,39,32,6,25,3,20,24,33,35,15,36,37,14,7,17,2,34]), index=None, placeholder="Select Product ID")

order_qty = st.selectbox("Order Quantity", [1,2,3,4,5,6,7,8], index=None, placeholder="Select Quantity")
discount = st.selectbox("Discount Applied", [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4], index=None, placeholder="Select Discount Rate")

unit_cost = st.number_input("Unit Cost", min_value=0.0, step=0.01)
unit_price = st.number_input("Unit Price", min_value=0.0, step=0.01)

order_date = st.date_input("Order Date")
ship_date = st.date_input("Ship Date")
delivery_date = st.date_input("Delivery Date")

sales_channel = st.selectbox("Sales Channel", ['Distributor', 'In-Store', 'Online', 'Wholesale'], index=None, placeholder="Select Sales Channel")
warehouse_code = st.selectbox("Warehouse Code", [
    'WARE-MKL1006', 'WARE-NBV1002', 'WARE-NMK1003', 'WARE-PUJ1005',
    'WARE-UHY1004', 'WARE-XYS1001'
], index=None, placeholder="Select Warehouse Code")

# --- Prediction Logic ---
if st.button("Predict Total Revenue"):
    # --- Basic Validation ---
    missing_fields = any(field is None for field in [
        sales_team_id, customer_id, store_id, product_id,
        order_qty, discount, sales_channel, warehouse_code
    ]) or unit_cost == 0.0 or unit_price == 0.0

    if missing_fields:
        st.warning("⚠️ Please make sure **all fields** are filled in and numeric values are non-zero.")
    elif ship_date < order_date or delivery_date < ship_date:
        st.error("🚫 Invalid date sequence. Ensure: Order Date ≤ Ship Date ≤ Delivery Date.")
    else:
        # --- Derived Date Features ---
        order_to_ship = (ship_date - order_date).days
        ship_to_delivery = (delivery_date - ship_date).days
        order_to_delivery = (delivery_date - order_date).days

        # --- Target Encoding ---
        def encode(mapping, key):
            return mapping.get(key, np.mean(list(mapping.values())))

        input_dict = {
            '_SalesTeamID_TE': encode(encoding_maps['_SalesTeamID'], sales_team_id),
            '_CustomerID_TE': encode(encoding_maps['_CustomerID'], customer_id),
            '_StoreID_TE': encode(encoding_maps['_StoreID'], store_id),
            '_ProductID_TE': encode(encoding_maps['_ProductID'], product_id),
            'Order Quantity': order_qty,
            'Discount Applied': discount,
            'Unit Cost': unit_cost,
            'Unit Price': unit_price,
            'Order_to_Ship_Days': order_to_ship,
            'Ship_to_Delivery_Days': ship_to_delivery,
            'Order_to_Delivery_Days': order_to_delivery,
            'Sales Channel_In-Store': 0,
            'Sales Channel_Online': 0,
            'Sales Channel_Wholesale': 0,
            'WarehouseCode_WARE-NBV1002': 0,
            'WarehouseCode_WARE-NMK1003': 0,
            'WarehouseCode_WARE-PUJ1005': 0,
            'WarehouseCode_WARE-UHY1004': 0,
            'WarehouseCode_WARE-XYS1001': 0,
        }

        # One-hot encode sales channel (Distributor is the base)
        if sales_channel != 'Distributor':
            input_dict[f'Sales Channel_{sales_channel}'] = 1

        # One-hot encode warehouse (WARE-MKL1006 is the base)
        if warehouse_code != 'WARE-MKL1006':
            input_dict[f'WarehouseCode_{warehouse_code}'] = 1

        # Ensure all expected features are present and ordered
        for col in expected_features:
            if col not in input_dict:
                input_dict[col] = 0  # Fill missing dummy columns with 0

        input_df = pd.DataFrame([input_dict])[expected_features]  # Reorder columns

        # --- Prediction ---
        prediction = model.predict(input_df)[0]

        if hasattr(model, "estimators_"):
            preds = np.array([est.predict(input_df)[0] for est in model.estimators_])
            std_dev = np.std(preds)
            st.success(f"💰 Predicted Total Revenue: **{prediction:.2f}**")
            st.info(f"📊 Confidence Interval: ±{std_dev:.2f}")
        else:
            st.success(f"💰 Predicted Total Revenue: **{prediction:.2f}**")
            st.warning("Confidence interval not available.")
