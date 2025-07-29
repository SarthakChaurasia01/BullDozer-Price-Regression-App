import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="🚜 Bulldozer Price Predictor", layout="centered")
st.title("🚜 Bulldozer Price Prediction App")
st.write("Enter bulldozer features to predict its auction sale price.")

# Load dataset

@st.cache_data
def load_data():
    return pd.read_csv("TrainAndValid.csv.zip", compression='zip')

df = load_data()


# Select features for a simple model
features = ['YearMade', 'MachineHoursCurrentMeter', 'ProductSize', 'Enclosure']
data = data[features + ['SalePrice']].dropna()

# Encode categorical features
data['ProductSize'] = data['ProductSize'].astype('category').cat.codes
data['Enclosure'] = data['Enclosure'].astype('category').cat.codes

X = data[features]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar form
st.sidebar.header("📝 Input Bulldozer Features")
with st.sidebar.form("input_form"):
    year_made = st.slider("Year Made", 1940, 2020, 2000)
    usage = st.slider("Machine Hours Used", 0, 100000, 5000)
    product_size = st.selectbox("Product Size", ['Mini', 'Small', 'Medium', 'Large', 'Extra Large', 'Large / Medium'])
    enclosure = st.selectbox("Enclosure Type", ['None or Unspecified', 'Open', 'EROPS', 'EROPS w AC', 'OROPS'])
    submitted = st.form_submit_button("Predict Price")

# Encode inputs
size_map = {'Mini': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Extra Large': 4, 'Large / Medium': 5}
enclosure_map = {'None or Unspecified': 0, 'Open': 1, 'EROPS': 2, 'EROPS w AC': 3, 'OROPS': 4}

if submitted:
    input_data = pd.DataFrame({
        'YearMade': [year_made],
        'MachineHoursCurrentMeter': [usage],
        'ProductSize': [size_map[product_size]],
        'Enclosure': [enclosure_map[enclosure]]
    })

    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Sale Price")
    st.success(f"${prediction:,.2f}")

    st.subheader("📉 Model RMSE on Test Set")
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    st.write(f"RMSE: ${rmse:,.2f}")
