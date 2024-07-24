import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO

# Load CSV data
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
try:
    data = pd.read_csv(csv_url)
    st.write("CSV data loaded successfully.")
except Exception as e:
    st.error(f"Error loading CSV data: {str(e)}")
    raise e

# Ensure the 'ultralytics' package is installed
try:
    import ultralytics
except ImportError:
    os.system("pip install ultralytics")
    import ultralytics

# Load the pre-trained YOLOv8 model
try:
    model = YOLO('yolov8x.pt')  # Using the more accurate model
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    raise e

# Function to preprocess image for YOLOv8
def preprocess_image(image):
    image = image.convert('RGB')
    image = np.array(image)
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

    # Initialize counters
    car_count = 0
    motorcycle_count = 0
    pedestrian_count = 0

    # Count the occurrences of each object type
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        if cls == 'car':
            car_count += 1
        elif cls == 'motorcycle':
            motorcycle_count += 1
        elif cls == 'person':
            pedestrian_count += 1

    # Display the results
    st.write(f"Cars: {car_count}")
    st.write(f"Motorcycles: {motorcycle_count}")
    st.write(f"Pedestrians: {pedestrian_count}")

    # Render and display the image with detections
    annotated_image = results[0].plot()
    st.image(annotated_image, caption='Detected Image', use_column_width=True)
