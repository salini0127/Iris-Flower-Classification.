import streamlit as st
from PIL import Image

# App title
st.title("Iris Flower Classification 📊")

# Load and display the saved plot
try:
    image = Image.open("iris_plot.png")  # Ensure correct path
    st.image(image, caption="Iris Dataset Plot", use_column_width=True)
except FileNotFoundError:
    st.warning("⚠️ Image file not found! Please check the file path.")

st.write("""
This visualization shows the distribution of different Iris flower species based on various features such as:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dataset consists of three species: **Setosa, Versicolor, and Virginica**.
""")

# Allow users to upload their own image
uploaded_image = st.file_uploader("Upload an Iris dataset visualization", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("✅ Image uploaded successfully!")

