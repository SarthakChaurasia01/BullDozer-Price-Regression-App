import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🚜 Bulldozer Price Prediction (Pre-trained Model)")

# ---------------------------
# 1. Load model and features
# ---------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("bulldozer_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")  # ✅ Correct file name
    return model, feature_columns

model, feature_columns = load_model_and_features()

# ---------------------------
# 2. Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {col: st.sidebar.number_input(col, value=0.0) for col in feature_columns}

# ---------------------------
# 3. Prediction
# ---------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
