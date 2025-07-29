import streamlit as st
import pandas as pd
import joblib

# Load the model and encoder
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("TrainAndValid.zip", compression='zip')

# Define the feature columns used in the model
features = ['YearMade', 'MachineHoursCurrentMeter', 'ModelID', 'ProductGroup', 'Enclosure']

# Page config
st.set_page_config(page_title="Bulldozer Price Prediction", layout="wide")
st.title("🚜 Bulldozer Price Predictor")
st.markdown("Predict the **sale price** of a bulldozer based on its characteristics.")

# Load model and data
model = load_model()
data = load_data()

# Filter data for relevant features
data = data[features + ['SalePrice']].dropna()

# Sidebar inputs
st.sidebar.header("Input Bulldozer Features")
user_input = {
    'YearMade': st.sidebar.slider('Year Made', int(data['YearMade'].min()), int(data['YearMade'].max()), 2000),
    'MachineHoursCurrentMeter': st.sidebar.slider('Machine Hours', 0, int(data['MachineHoursCurrentMeter'].max()), 1000),
    'ModelID': st.sidebar.selectbox('Model ID', sorted(data['ModelID'].unique())),
    'ProductGroup': st.sidebar.selectbox('Product Group', sorted(data['ProductGroup'].unique())),
    'Enclosure': st.sidebar.selectbox('Enclosure', sorted(data['Enclosure'].unique())),
}

# Convert input into DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict Sale Price"):
    # Align columns and dtypes with training data if needed
    for col in ['ProductGroup', 'Enclosure']:
        input_df[col] = input_df[col].astype('category')

    prediction = model.predict(input_df)[0]
    st.subheader(f"💰 Predicted Sale Price: ${prediction:,.2f}")

# Show some raw data if user checks box
if st.checkbox("Show sample data"):
    st.write(data.sample(5))
