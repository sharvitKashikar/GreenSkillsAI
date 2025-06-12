import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Salary Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #3c3c3c;
    }
    .css-1d391kg {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.title("📈 Salary Data Visualizer")
st.subheader("Explore salary trends based on experience")

# Sidebar for user info or customization
st.sidebar.title("Settings")
st.sidebar.markdown("Use this app to view and explore salary data.")

# Load data
try:
    df = pd.read_csv(r"C:\Users\priya\OneDrive\Desktop\greenAI\day-6\machine learning\Salary_Data.csv")
    st.success("Data loaded successfully!")

    with st.expander("📋 View Raw Data"):
        st.dataframe(df, use_container_width=True)

    # Plotting
    st.markdown("### 📉 Salary vs Experience Line Chart (Interactive)")
    fig = px.line(df, x=df.columns[0], y=df.columns[1], markers=True,
                  labels={df.columns[0]: "Years of Experience", df.columns[1]: "Salary"},
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
