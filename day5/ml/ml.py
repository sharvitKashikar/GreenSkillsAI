import streamlit as st
import pandas as pd

st.write("Hello this is my first streanlit app")
df = pd.read_csv(r"/Users/sharvitkashikar/Training-june/day5/ml/Salary_Data.csv")
st.write(df)
st.line_chart(df)