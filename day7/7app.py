import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('/Users/sharvitkashikar/Training-june/day7/ecosystem_health_model.pkl')

# Title
st.title("Ecosystem Health Prediction")

# User input fields
st.header("Enter Environmental Parameters:")
wq = st.number_input("Water Quality Index", min_value=0.0, max_value=100.0, value=50.0)
aqi = st.number_input("Air Quality Index", min_value=0.0, max_value=500.0, value=100.0)
biodiversity = st.number_input("Biodiversity Index", min_value=0.0, max_value=1.0, value=0.5)
veg_cover = st.number_input("Vegetation Cover Index", min_value=0.0, max_value=100.0, value=50.0)
soil_ph = st.number_input("Soil pH Index", min_value=0.0, max_value=14.0, value=7.0)

# Predict button
if st.button("Predict Ecosystem Health"):
    user_input = np.array([[wq, aqi, biodiversity, veg_cover, soil_ph]])
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.success("Predicted Ecosystem Health: **Healthy** 🌱")
    elif prediction == 0:
        st.warning("Predicted Ecosystem Health: **At Risk** ⚠️")
    elif prediction == 2:
        st.error("Predicted Ecosystem Health: **Degraded** ❌")
    else:
        st.info("Unknown prediction.")

st.markdown("---")
st.caption("Model: Gaussian Naive Bayes | Data: Environmental Indices")