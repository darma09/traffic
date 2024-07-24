import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load CSV data from GitHub
csv_url = "https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv"
data = pd.read_csv(csv_url)

# Path to the model
model_path = "traffic_cnn_model.h5"

# Function to load the model
def load_cnn_model(path):
    try:
        model = load_model(path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the pre-trained CNN model
model = load_cnn_model(model_path)

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app title
st.title("Traffic Analysis and Image Upload")

# Display the CSV data
st.subheader("Traffic Data")
st.write(data.head())

# Image upload functionality
st.subheader("Upload Traffic Images")
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        if model:
            # Preprocess the image
            processed_image = preprocess_image(img)
            
            # Perform image analysis using the CNN model
            prediction = model.predict(processed_image)
            st.write(f"Prediction: {np.argmax(prediction)}")

# Basic data analysis
st.subheader("Basic Data Analysis")
st.write("Descriptive Statistics of Traffic Volume")
st.write(data.describe())

# Show some plots based on CSV data
st.subheader("Traffic Volume Over Time")
st.line_chart(data['traffic_volume'])

# Run the app
if __name__ == '__main__':
    st.run()
