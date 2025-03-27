import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('decision_tree_model.pkl')  # Load Decision Tree model
scaler = joblib.load('scaler.pkl')  # Load scaler

st.title("🌸 Iris Flower Classification")
st.write("Enter the flower's features to predict the species.")

# User input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"🌿 Predicted Species: {species[int(prediction[0])]} 🌿")
    import streamlit as st
from PIL import Image

st.title("Iris Flower Classification 📊")  # App title

# Load and display the saved plot
image = Image.open("iris_plot.png")
st.image(image, caption="Iris Dataset Plot", use_column_width=True)

st.write("This plot visualizes the Iris dataset.")  # Add some description
