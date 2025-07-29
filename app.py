import streamlit as st
import pandas as pd
import gdown
import zipfile
from fastai.tabular.all import load_learner

# ======================
# CONFIGURATION
# ======================
DATA_URL = "https://drive.google.com/uc?id=YOUR_DATASET_FILE_ID"  # Replace with your file ID
MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"   # Replace with your model file ID

DATA_ZIP = "TrainAndValid.zip"
MODEL_FILE = "bulldozer_model.pkl"

# ======================
# DOWNLOAD AND EXTRACT
# ======================
@st.cache_resource
def download_and_prepare_files():
    # Download dataset if not already present
    if not st.session_state.get("data_ready", False):
        st.write("Downloading dataset...")
        gdown.download(DATA_URL, DATA_ZIP, quiet=False)
        with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall("data")
        st.session_state["data_ready"] = True

    # Download model if not already present
    if not st.session_state.get("model_ready", False):
        st.write("Downloading model...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
        st.session_state["model_ready"] = True

    return load_learner(MODEL_FILE)

# ======================
# LOAD MODEL
# ======================
model = download_and_prepare_files()

# ======================
# STREAMLIT APP UI
# ======================
st.title("🚜 Bulldozer Price Prediction")

st.markdown("Upload details of the bulldozer to predict its selling price.")

year_made = st.number_input("Year Made", min_value=1900, max_value=2025, value=2010)
sale_month = st.selectbox("Sale Month", list(range(1, 13)))
sale_year = st.number_input("Sale Year", min_value=2000, max_value=2025, value=2012)
machine_hours = st.number_input("Machine Hours", min_value=0, value=5000)

if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "YearMade": year_made,
        "saleMonth": sale_month,
        "saleYear": sale_year,
        "MachineHoursCurrentMeter": machine_hours
    }])

    prediction = model.predict(input_data)
    st.success(f"Estimated Bulldozer Price: ${prediction[0]:,.2f}")
