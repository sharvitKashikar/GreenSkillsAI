import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
appliance_data = pd.read_csv(r'C:\Users\priya\OneDrive\Desktop\greenAI\day-7\sample streamlit app\appliance_energy.csv')

# Feature and target
x = appliance_data[['Temperature (°C)']]
y = appliance_data[['Energy Consumption (kWh)']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title('Simple Linear Regression on Appliance Energy Consumption')
st.write('Mean Squared Error:', mse)
st.write('R² Score:', r2)

# Plotting
fig, ax = plt.subplots()
ax.scatter(x_test, y_test, color='blue', label='Actual Data')
ax.plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Energy Consumption (kWh)')
ax.set_title('Regression Model')
ax.legend()
st.pyplot(fig)

# User input
st.subheader("Predict Energy Consumption")
temperature_input = st.number_input("Enter Temperature (°C):", format="%.2f")
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = []

if st.button("Predict"):
    prediction = model.predict([[temperature_input]])[0][0]
    st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
    st.session_state.prediction_data.append({
        "Temperature (°C)": temperature_input,
        "Predicted Energy Consumption (kWh)": round(prediction, 2)
    })
#save the model
import joblib
joblib.dump(model, r'C:\Users\priya\OneDrive\Desktop\greenAI\day-7\sample streamlit app\linear_regression_model.pkl')

# Display predictions
if st.session_state.prediction_data:
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_data)
    st.table(history_df)

