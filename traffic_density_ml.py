import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch

# Load CSV data from GitHub
csv_url = "https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv"
data = pd.read_csv(csv_url)

# Load the pre-trained YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to preprocess image for YOLOv5
def preprocess_image(image):
    image = np.array(image)
    return image

# Colors for different classes
COLORS = {
    'car': (0, 255, 0),        # Green
    'motorcycle': (0, 0, 255), # Red
    'bus': (255, 0, 0),        # Blue
    'truck': (255, 255, 0),    # Cyan
    'person': (255, 0, 255)    # Magenta
}

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

        # Preprocess the image
        processed_image = preprocess_image(img)
        
        # Perform object detection using YOLOv5
        results = model(processed_image)
        df = results.pandas().xyxy[0]

        # Draw bounding boxes and labels on the image
        detected_people = 0
        for _, row in df.iterrows():
            label = row['name']
            confidence = row['confidence'] * 100  # Convert to percentage
            if confidence < 50:  # Only consider detections with confidence above 50%
                continue
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = COLORS.get(label, (255, 255, 255)) # Default to white if the class is not in COLORS
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_image, f'{label} {confidence:.0f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if label == 'person' and not any(row['name'] == 'motorcycle' for _, row in df.iterrows() if x1 >= int(row['xmin']) and x2 <= int(row['xmax']) and y1 >= int(row['ymin']) and y2 <= int(row['ymax'])):
                detected_people += 1

        # Count vehicles
        vehicle_counts = df['name'].value_counts()
        num_cars = vehicle_counts.get('car', 0)
        num_motorcycles = vehicle_counts.get('motorcycle', 0)
        st.write(f"Number of cars: {num_cars}")
        st.write(f"Number of motorcycles: {num_motorcycles}")
        st.write(f"Number of people: {detected_people}")
        
        # Convert image back to PIL format for displaying
        processed_image = Image.fromarray(processed_image)
        st.image(processed_image, caption='Processed Image with Bounding Boxes', use_column_width=True)

        # Show detailed results
        st.write(df)

# Basic data analysis
st.subheader("Basic Data Analysis")
st.write("Descriptive Statistics of Traffic Volume")
st.write(data.describe())

# Show some plots based on CSV data
st.subheader("Traffic Volume Over Time")
st.line_chart(data['traffic_volume'])
