import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb



df = pd.read_csv(r"/Users/sharvitkashikar/Training-june/day5/ml/Salary_Data.csv")
st.title("Salary Prediction App")
exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=1, step=1)
if(st.button("Predict")):
    model = jb.load(r"/Users/sharvitkashikar/Training-june/day5/ml/salary_model.pkl")
    pred = model.predict(np.array([[exp]]))
    st.success(f"Predicted Salary is {pred[0]:.2f} $")
