import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# -------------------
# Page Configuration
# -------------------
st.set_page_config(page_title="Bulldozer Price Predictor", layout="wide")
st.title("🚜 Bulldozer Price Predictor")
st.markdown("Predict the **sale price** of a bulldozer based on its features.")

# -------------------
# Load Dataset
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TrainAndValid.zip", compression='zip', low_memory=False)
    return df

data = load_data()

# -------------------
# Feature Selection
# -------------------
features = ['YearMade', 'MachineHoursCurrentMeter', 'ModelID', 'ProductGroup', 'Enclosure']
target = 'SalePrice'

df = data[features + [target]].dropna()

# -------------------
# Encoding Categorical Features
# -------------------
@st.cache_resource
def train_model(df):
    X = df[features]
    y = df[target]

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, encoder

model, encoder = train_model(df)

# -------------------
# Sidebar Inputs
# -------------------
st.sidebar.header("Input Bulldozer Features")

user_input = {
    'YearMade': st.sidebar.slider('Year Made', int(df['YearMade'].min()), int(df['YearMade'].max()), 2000),
    'MachineHoursCurrentMeter': st.sidebar.slider('Machine Hours', 0, int(df['MachineHoursCurrentMeter'].max()), 1000),
    'ModelID': st.sidebar.selectbox('Model ID', sorted(df['ModelID'].unique())),
    'ProductGroup': st.sidebar.selectbox('Product Group', sorted(df['ProductGroup'].unique())),
    'Enclosure': st.sidebar.selectbox('Enclosure', sorted(df['Enclosure'].unique())),
}

input_df = pd.DataFrame([user_input])

# Encode categorical inputs
cat_features = ['ProductGroup', 'Enclosure']
input_df[cat_features] = encoder.transform(input_df[cat_features])

# -------------------
# Prediction
# -------------------
if st.button("Predict Sale Price"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"💰 Predicted Sale Price: ${prediction:,.2f}")

# -------------------
# Show Data Preview
# -------------------
if st.checkbox("Show raw sample data"):
    st.dataframe(df.sample(5))
