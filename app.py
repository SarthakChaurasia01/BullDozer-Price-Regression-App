import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("🚜 Bulldozer Price Prediction")

# ---------------------------
# 1. Download and Extract Dataset
# ---------------------------
zip_url = "https://your-file-link.com/TrainAndValid.zip"  # <-- upload your zip somewhere and put link
zip_path = "TrainAndValid.zip"
extract_dir = "dataset"

if not os.path.exists(extract_dir):
    st.write("📥 Downloading dataset...")
    import requests
    r = requests.get(zip_url)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    st.write("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

st.success("✅ Dataset ready!")

# ---------------------------
# 2. Load Dataset
# ---------------------------
csv_path = os.path.join(extract_dir, "TrainAndValid.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    st.write("### Dataset Preview", df.head())
else:
    st.error("❌ TrainAndValid.csv not found in the extracted ZIP.")
    st.stop()

# ---------------------------
# 3. Train Model (Sample to Avoid Timeout)
# ---------------------------
@st.cache_resource
def train_model(data):
    st.write("Training model...")
    if 'SalePrice' not in data.columns:
        st.error("The dataset must contain a 'SalePrice' column.")
        return None

    # Use a sample to speed up training
    data = data.sample(50000, random_state=42) if len(data) > 50000 else data

    X = data.drop("SalePrice", axis=1).select_dtypes(include=[np.number]).fillna(0)
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

with st.spinner("⏳ Training model..."):
    model, feature_columns = train_model(df)

# ---------------------------
# 4. User Input
# ---------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {col: st.sidebar.number_input(col, value=0.0) for col in feature_columns}

# ---------------------------
# 5. Prediction
# ---------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
