import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, features, and label mapping
model = joblib.load('/Users/sharvitkashikar/Training-june/day6/DT_RF_LR/best_weather_model.pkl')
features = joblib.load('/Users/sharvitkashikar/Training-june/day6/DT_RF_LR/model_features.pkl')
label_mapping = joblib.load('/Users/sharvitkashikar/Training-june/day6/DT_RF_LR/label_mapping.pkl')

st.title("Weather Prediction App")

# Input fields for each feature
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("Predict Weather"):
    pred_code = model.predict(input_df)[0]
    pred_label = label_mapping[pred_code]
    st.success(f"Predicted Weather: {pred_label}")