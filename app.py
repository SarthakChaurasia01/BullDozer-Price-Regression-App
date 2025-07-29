import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import re

st.title("🚜 Bulldozer Price Prediction")

# ---------------------------
# 1. Google Drive Link Input
# ---------------------------
st.sidebar.header("Dataset Settings")
drive_link = st.sidebar.text_input(
    "Google Drive CSV Link",
    value="https://drive.google.com/file/d/1hwVrEAaYGV_aJBMhZ9inV6MX-Am2fM4u/view?usp=drive_link"
)

# Extract file ID from link
match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
if match:
    file_id = match.group(1)
    csv_path = "TrainAndValid.csv"
    file_url = f"https://drive.google.com/uc?id={file_id}"
else:
    st.error("❌ Invalid Google Drive link format.")
    st.stop()

# ---------------------------
# 2. Download CSV via gdown (handles large files)
# ---------------------------
if not st.session_state.get("data_loaded"):
    st.write("📥 Downloading dataset from Google Drive...")
    gdown.download(file_url, csv_path, quiet=False)
    st.session_state["data_loaded"] = True

st.success("✅ Dataset downloaded successfully!")

# ---------------------------
# 3. Load Dataset
# ---------------------------
try:
    df = pd.read_csv(csv_path, low_memory=False)
except pd.errors.ParserError:
    df = pd.read_csv(csv_path, delimiter=";", low_memory=False)

st.write("### Dataset Preview", df.head())

# ---------------------------
# 4. Train Model (Sample to Speed Up)
# ---------------------------
@st.cache_resource
def train_model(data):
    if 'SalePrice' not in data.columns:
        st.error("Dataset must contain 'SalePrice' column.")
        return None

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
# 5. User Input
# ---------------------------
st.sidebar.header("Enter Bulldozer Details")
input_data = {col: st.sidebar.number_input(col, value=0.0) for col in feature_columns}

# ---------------------------
# 6. Prediction
# ---------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Bulldozer Price: ${prediction[0]:,.2f}")
