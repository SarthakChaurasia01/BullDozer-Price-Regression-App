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
    feature_columns = joblib.load("feature_columns.pkl")  # these are final columns after preprocessing
    return model, feature_columns

model, feature_columns = load_model_and_features()

# ---------------------------
# 2. Load dataset (for reference)
# ---------------------------
df = pd.read_csv("TrainAndValid.csv", low_memory=False)

# Determine which features are categorical (before encoding)
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

# ---------------------------
# 3. Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {}

for col in categorical_features:
    options = df[col].dropna().unique().tolist()
    input_data[col] = st.sidebar.selectbox(col, options)

numeric_features = [f for f in df.columns if f not in categorical_features and f in feature_columns]
for col in numeric_features:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# ---------------------------
# 4. Preprocessing (Align with Training Features)
# ---------------------------
# Convert categorical to one-hot
input_df = pd.get_dummies(input_df)

# Reindex to match training columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ---------------------------
# 5. Prediction
# ---------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
