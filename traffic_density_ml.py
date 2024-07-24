import streamlit as st
import torch
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import os

# Load CSV data
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/traffics.csv'
data = pd.read_csv(csv_url)

# Ensure the 'ultralytics' package is installed
try:
    import ultralytics
except ImportError:
    os.system("pip install ultralytics")
    import ultralytics

# Load the pre-trained YOLOv5 model from PyTorch Hub
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Using a more accurate model
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    raise e

# Function to preprocess image for YOLOv5
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((640, 640))  # Resize image to 640x640 pixels
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # Normalize image
    return image

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess and make predictions
    processed_image = preprocess_image(image)
    results = model(processed_image)

    # Convert results to pandas dataframe
    df = results.pandas().xyxy[0]

    # Draw bounding boxes and labels on image
    for _, row in df.iterrows():
        st.write(f"{row['name']}: {row['confidence']*100:.2f}%")

    # Display results
    st.image(results.render(), caption='Detected Image', use_column_width=True)
