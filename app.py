import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# 1. App Title
# ---------------------------
st.title("🚜 Bulldozer Price Prediction")

# ---------------------------
# 2. Extract ZIP file
# ---------------------------
zip_path = "TrainAndValid.zip"  # The zip file should be in the same repo
extract_dir = "dataset"

if not os.path.exists(extract_dir):
    st.write("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

st.success("✅ Dataset ready!")

# ---------------------------
# 3. Load CSV
# ---------------------------
csv_path = os.path.join(extract_dir, "TrainAndValid.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    st.write("### Dataset Preview", df.head())
else:
    st.error("❌ TrainAndValid.csv not found inside ZIP!")

# ---------------------------
# 4. Train or Cache Model
# ---------------------------
@st.cache_resource
def train_model(data):
    if 'SalePrice' not in data.columns:
        st.error("The dataset must contain a 'SalePrice' column for training.")
        return None

    X = data.drop("SalePrice", axis=1).select_dtypes(include=[np.number]).fillna(0)
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

model, feature_columns = train_model(df)

# ---------------------------
# 5. User Input
# ---------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {}

for feature in feature_columns:
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# ---------------------------
# 6. Prediction
# ---------------------------
if st.button("Predict Price"):
    if model is not None:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
    else:
        st.error("Model training failed.")
