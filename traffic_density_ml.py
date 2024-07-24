import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os
import cv2

# Ensure OpenCV is installed correctly
try:
    import cv2
except ImportError as e:
    st.error("OpenCV is not installed. Please install it using 'pip install opencv-python-headless'")
    raise e

# Load CSV data from GitHub
csv_url = "https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv"
data = pd.read_csv(csv_url)

# Ensure ultralytics is installed correctly
try:
    import ultralytics
except ImportError as e:
    st.error("Ultralytics is not installed. Please install it using 'pip install ultralytics'")
    raise e

# Load the YOLOv5n-seg model locally
model_path = Path("yolov5n-seg.pt")
if not model_path.exists():
    st.error("Model file yolov5n-seg.pt not found. Please download it from the YOLOv5 repository and place it in the project directory.")
else:
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        raise e

# Function to preprocess image for YOLOv5
def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = torch.from_numpy(image).float()  # Convert to tensor
    return image

# Colors for different classes
COLORS = {
    'car': (0, 255, 0),          # Green
    'motorcycle': (0, 0, 255),   # Red
    'bus': (255, 0, 0),          # Blue
    'truck': (255, 255, 0),      # Cyan
    'person': (255, 0, 255),     # Magenta
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
        results = model(processed_image)  # Removed size argument
        df = results.pandas().xyxy[0]

        # Draw bounding boxes and labels on the image
        processed_image = processed_image.squeeze().permute(1, 2, 0).numpy()  # Convert back to original shape
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        detected_people = 0
        motorcycles = []

        for _, row in df.iterrows():
            if row['name'] == 'motorcycle':
                motorcycles.append((row['xmin'], row['ymin'], row['xmax'], row['ymax']))
            label = row['name']
            confidence = row['confidence'] * 100  # Convert to percentage
            if confidence < 50:  # Only consider detections with confidence above 50%
                continue
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            color = COLORS.get(label, (255, 255, 255))  # Default to white if the class is not in COLORS
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_image, f'{label} {confidence:.0f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if label == 'person':
                # Check if the person is on a motorcycle
                on_motorcycle = False
                for mx1, my1, mx2, my2 in motorcycles:
                    if (mx1 <= (x1 + x2) / 2 <= mx2) and (my1 <= (y1 + y2) / 2 <= my2):
                        on_motorcycle = True
                        break
                if not on_motorcycle:
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

# Basic data analysis
st.subheader("Basic Data Analysis")
st.write("Descriptive Statistics of Traffic Volume")
st.write(data.describe())

# Show some plots based on CSV data
st.subheader("Traffic Volume Over Time")
st.line_chart(data['traffic_volume'])
