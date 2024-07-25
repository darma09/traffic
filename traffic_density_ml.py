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
    
    # Count specific objects and filter detections
    labels = results.names
    counts = {"car": 0, "motorcycle": 0, "person": 0}
    filtered_detections = []
    for pred in results.pred[0]:
        label = labels[int(pred[-1])]
        if label == "motorcycle":
            counts["motorcycle"] += 1
            filtered_detections.append(pred)
        elif label == "car":
            counts["car"] += 1
            filtered_detections.append(pred)
        elif label == "person":
            # Check if the person is on a motorcycle by proximity (simplified logic)
            person_bbox = pred[:4]
            is_on_motorcycle = False
            for other_pred in results.pred[0]:
                other_label = labels[int(other_pred[-1])]
                if other_label == "motorcycle":
                    motorcycle_bbox = other_pred[:4]
                    if is_near(person_bbox, motorcycle_bbox):
                        is_on_motorcycle = True
                        break
            if not is_on_motorcycle:
                counts["person"] += 1
                filtered_detections.append(pred)

    # Draw boxes for filtered detections
    for det in filtered_detections:
        label = labels[int(det[-1])]
        if label in ["car", "motorcycle"]:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_image, counts

def is_near(bbox1, bbox2, threshold=0.5):
    """
    Determine if bbox1 is near bbox2 based on intersection over union (IoU) threshold.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)
    
    if inter_min_x < inter_max_x and inter_min_y < inter_max_y:
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
        return iou > threshold
    return False

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
