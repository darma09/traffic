import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import requests
import io

# Function to load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

@st.cache_resource
def load_data_from_url(url):
    response = requests.get(url)
    drive_file = io.StringIO(response.content.decode('utf-8'))
    data = pd.read_csv(drive_file)
    return data

def process_image(uploaded_file, model):
    image = Image.open(uploaded_file)
    results = model(image)
    results.render()  # updates results.imgs with boxes and labels
    
    # Convert result image to display in streamlit
    result_image = Image.fromarray(results.ims[0])  # Note: 'ims' attribute used here
    return result_image

# Streamlit app
st.title("Data Analysis and Object Detection App")

# Step 1: Download CSV file
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
data = load_data_from_url(csv_url)
st.write("CSV Data Analysis")
st.write(data.head())

# Step 3: Display CSV file in Streamlit
st.dataframe(data)

# Step 4: File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Processing image...")
    
    # Load YOLOv5 model
    model = load_yolo_model()

    # Step 5: Analyze the uploaded image
    result_image = process_image(uploaded_file, model)

    # Step 6: Display the result of image analysis
    st.image(result_image, caption='Processed Image', use_column_width=True)
else:
    st.write("Please upload an image file.")
