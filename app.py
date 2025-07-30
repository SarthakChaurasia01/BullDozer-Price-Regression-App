import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re

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
# 2. Google Drive CSV Downloader
# --------------------------------------
@st.cache_data
def download_csv(file_id):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None

    # Handle confirmation token for large files
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            tok
