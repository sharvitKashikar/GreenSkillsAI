import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare dataset
def train_and_save_model():
    data = pd.read_csv(r'C:\Users\priya\OneDrive\Desktop\greenAI\day-8\naive_bayes\ecosystem_data.csv')
    
    # Encode target variable
    data['ecosystem_health'] = data['ecosystem_health'].map({'healthy': 1, 'at risk': 0, 'degraded': 2})

    X = data[['water_quality', 'air_quality_index', 'biodiversity_index', 'vegetation_cover', 'soil_ph']]
    y = data['ecosystem_health']

    # Train-test split before SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Original training class distribution:", y_train.value_counts().to_dict())
    print("After SMOTE:", pd.Series(y_train_resampled).value_counts().to_dict())

    # Train model
    model = GaussianNB()
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save model
    joblib.dump(model, 'ecosystem_health_model.pkl')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'At Risk', 'Degraded'],
                yticklabels=['Healthy', 'At Risk', 'Degraded'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Run this once to train and save the model
train_and_save_model()
# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('ecosystem_health_model.pkl')

st.title("🌿 Ecosystem Health Prediction")
st.write("Enter the environmental indicators to predict ecosystem health:")

# Input fields
wq = st.number_input("Water Quality (0–100)", min_value=0.0, max_value=100.0, step=1.0)
aq = st.number_input("Air Quality Index (0–300)", min_value=0.0, max_value=300.0, step=1.0)
bi = st.number_input("Biodiversity Index (0–1)", min_value=0.0, max_value=1.0, step=0.01)
vc = st.number_input("Vegetation Cover (%)", min_value=0.0, max_value=100.0, step=1.0)
sp = st.number_input("Soil pH (0–14)", min_value=0.0, max_value=14.0, step=0.1)

if st.button("Predict Ecosystem Health"):
    input_data = np.array([[wq, aq, bi, vc, sp]])
    prediction = model.predict(input_data)

    health_status = {1: '🌱 Healthy', 0: '⚠️ At Risk', 2: '🛑 Degraded'}
    st.success(f"Predicted Ecosystem Health: **{health_status[prediction[0]]}**")
