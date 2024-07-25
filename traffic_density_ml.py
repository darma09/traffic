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
    
    # Count specific objects
    labels = results.names
    counts = {"car": 0, "motorcycle": 0, "person": 0}
    boxes = []

    for pred in results.pred[0]:
        label = labels[int(pred[-1])]
        if label == "motorcycle":
            # Check if there's a person riding the motorcycle
            motorcycle_box = pred[:4]
            for person_pred in results.pred[0]:
                if labels[int(person_pred[-1])] == "person":
                    person_box = person_pred[:4]
                    iou = calculate_iou(motorcycle_box, person_box)
                    if iou > 0.5:  # Assuming a threshold of 0.5 for IOU
                        counts["motorcycle"] += 1
                        boxes.append(motorcycle_box)
                        break
            else:
                counts["motorcycle"] += 1
                boxes.append(motorcycle_box)
        elif label == "car":
            counts["car"] += 1
            boxes.append(pred[:4])
        elif label == "person":
            counts["person"] += 1
            boxes.append(pred[:4])

    draw_boxes(image, boxes)
    
    return result_image, counts

def calculate_iou(box1, box2):
    # Calculate intersection over union (IOU) between two boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(xi2 - xi1 + 1, 0) * max(yi2 - yi1 + 1, 0)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=2)
    return image

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
