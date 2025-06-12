import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

# Load data
data = pd.read_csv(r"/Users/sharvitkashikar/Training-june/day6/appliance_energy (1).csv")
st.write("Dataset")
st.write(data)

# features and target
features = ['Temperature (°C)']
target = 'Energy Consumption (kWh)'

# Split data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"R-squared Score: {r2:.4f}")
st.write(f"Mean Squared Error: {mse:.4f}")

# input for prediction
st.subheader("Input Features")
input_data = []
for col in features:
    val = st.number_input(f"Enter value for {col}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f} kWh")