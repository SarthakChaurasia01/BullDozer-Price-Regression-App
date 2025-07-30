import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🚜 Bulldozer Price Prediction (Pre-trained Model)")

# --------------------------------------
# 1. Load pre-trained model and features
# --------------------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("bulldozer_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model_and_features()

# --------------------------------------
# 2. Load Test CSV (from GitHub repo)
# --------------------------------------
@st.cache_data
def load_test_data():
    return pd.read_csv("Test.csv", low_memory=False)

df = load_test_data()
st.success(f"✅ Test dataset loaded! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# --------------------------------------
# 3. Identify categorical and numeric features
# --------------------------------------
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
numeric_features = [f for f in df.columns if f not in categorical_features and f in feature_columns]

# --------------------------------------
# 4. Sidebar user input
# --------------------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {}

# Categorical fields
for col in categorical_features:
    options = df[col].dropna().unique().tolist()
    if len(options) > 50:
        input_data[col] = st.sidebar.text_input(col, value=str(options[0]))
    else:
        input_data[col] = st.sidebar.selectbox(col, options)

# Numeric fields
for col in numeric_features:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# --------------------------------------
# 5. Preprocess input
# --------------------------------------
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# --------------------------------------
# 6. Prediction
# --------------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
