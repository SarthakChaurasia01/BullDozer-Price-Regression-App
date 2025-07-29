import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("🚜 Bulldozer Price Prediction")

# ---------------------------
# 1. Download CSV from Google Drive
# ---------------------------
csv_url = "https://drive.google.com/file/d/1Id_p4lhVDanRg8y3jMZdLeTlIWDpzS1a/view?usp=sharing"  # Replace with your file ID
csv_path = "TrainAndValid.csv"

if not st.session_state.get("data_loaded"):
    st.write("📥 Downloading dataset...")
    r = requests.get(csv_url)
    with open(csv_path, "wb") as f:
        f.write(r.content)
    st.session_state["data_loaded"] = True

st.success("✅ Dataset ready!")

# ---------------------------
# 2. Load Dataset
# ---------------------------
df = pd.read_csv(csv_path, low_memory=False)
st.write("### Dataset Preview", df.head())

# ---------------------------
# 3. Train Model (Sample to Speed Up)
# ---------------------------
@st.cache_resource
def train_model(data):
    if 'SalePrice' not in data.columns:
        st.error("Dataset must contain 'SalePrice' column.")
        return None

    # Sample to speed up training if very large
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
