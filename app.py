import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re

st.title("🚜 Bulldozer Price Prediction (Pre-trained Model)")

# ---------------------------
# 1. Load model and features
# ---------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("bulldozer_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model_and_features()

# ---------------------------
# 2. Google Drive CSV Download
# ---------------------------
drive_link = st.sidebar.text_input(
    "Google Drive CSV Link",
    value="https://drive.google.com/file/d/1hwVrEAaYGV_aJBMhZ9inV6MX-Am2fM4u/view?usp=drive_link"
)

match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
if match:
    file_id = match.group(1)
    csv_url = f"https://drive.google.com/uc?export=download&id={file_id}"
else:
    st.error("❌ Invalid Google Drive link.")
    st.stop()

@st.cache_data
def download_csv(url):
    return pd.read_csv(url, low_memory=False)

st.write("📥 Loading dataset from Google Drive...")
df = download_csv(csv_url)
st.success("✅ Dataset loaded successfully!")

# ---------------------------
# 3. Identify categorical features
# ---------------------------
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

# ---------------------------
# 4. Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {}

for col in categorical_features:
    options = df[col].dropna().unique().tolist()
    input_data[col] = st.sidebar.selectbox(col, options)

numeric_features = [f for f in df.columns if f not in categorical_features and f in feature_columns]
for col in numeric_features:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# ---------------------------
# 5. Preprocess input
# ---------------------------
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ---------------------------
# 6. Prediction
# ---------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
