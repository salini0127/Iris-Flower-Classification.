import streamlit as st
import numpy as np
import joblib

# Load the trained model
try:
    model = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("⚠️ Model or scaler file not found! Please upload them.")

# App Title
st.title("🌸 Iris Flower Classification")

# User Input Fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction Button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌿 Predicted Species: {species[int(prediction[0])]} 🌿")
