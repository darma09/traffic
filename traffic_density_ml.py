import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import requests
import io
import cv2

# Function to load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)  # Using YOLOv5x for better accuracy
    return model

@st.cache_resource
def load_data_from_url(url):
    response = requests.get(url)
    drive_file = io.StringIO(response.content.decode('utf-8'))
    data = pd.read_csv(drive_file)
    return data

def preprocess_image(image):
    # Convert image to RGB if it's in a different mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image to a standard size to handle different resolutions
    image = image.resize((640, 640))
    return image

def process_image(uploaded_file, model):
    image = Image.open(uploaded_file)
    image = preprocess_image(image)
    results = model(image)
    results.render()  # updates results.imgs with boxes and labels

    # Convert result image to display in streamlit
    result_image = Image.fromarray(results.ims[0])
    
    # Count specific objects and remove persons riding motorcycles
    labels = results.names
    boxes = results.xyxy[0].cpu().numpy()
    
    counts = {"car": 0, "motorcycle": 0, "person": 0}
    person_boxes = []
    motorcycle_boxes = []

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(boxes):
        label = labels[int(cls)]
        if label == "person":
            person_boxes.append((x1, y1, x2, y2))
        elif label == "motorcycle":
            motorcycle_boxes.append((x1, y1, x2, y2))
            counts['motorcycle'] += 1

    # Check if persons are within the motorcycle bounding boxes
    for (px1, py1, px2, py2) in person_boxes:
        for (mx1, my1, mx2, my2) in motorcycle_boxes:
            if px1 >= mx1 and py1 >= my1 and px2 <= mx2 and py2 <= my2:
                break
        else:
            counts['person'] += 1

    # Draw bounding boxes for motorcycles only
    result_image_np = np.array(result_image)
    for (mx1, my1, mx2, my2) in motorcycle_boxes:
        result_image_np = cv2.rectangle(result_image_np, (int(mx1), int(my1)), (int(mx2), int(my2)), (0, 255, 0), 2)
        result_image_np = cv2.putText(result_image_np, "motorcycle", (int(mx1), int(my1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_image = Image.fromarray(result_image_np)

    return result_image, counts

# Streamlit app
st.title("Data Analysis and Object Detection App")

# Step 1: Download CSV file
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
data = load_data_from_url(csv_url)
st.write("CSV Data Analysis")
st.write(data.head())

# Step 3: Display CSV file in Streamlit
st.dataframe(data)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Analyze the uploaded image
if uploaded_file is not None:
    st.write("Processing image...")

    # Load YOLOv5 model
    model = load_yolo_model()

    # Process the uploaded image
    result_image, counts = process_image(uploaded_file, model)

    # Display the result of image analysis
    st.image(result_image, caption='Processed Image', use_column_width=True)

    # Display the counts of cars, motorcycles, and pedestrians
    st.write(f"Cars detected: {counts['car']}")
    st.write(f"Motorcycles detected: {counts['motorcycle']}")
    st.write(f"Pedestrians detected: {counts['person']}")
else:
    st.write("Please upload an image file.")
