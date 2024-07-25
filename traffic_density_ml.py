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

def calculate_iou(box1, box2):
    # Calculate the intersection over union (IoU) of two bounding boxes.
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    # Determine the coordinates of the intersection rectangle
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    # Calculate area of the intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    
    # Calculate the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - inter_area
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def refine_person_count(person_boxes, motorcycle_boxes, iou_threshold):
    refined_person_count = 0
    for (px1, py1, px2, py2) in person_boxes:
        is_riding = False
        for (mx1, my1, mx2, my2) in motorcycle_boxes:
            iou = calculate_iou((px1, py1, px2, py2), (mx1, my1, mx2, my2))
            if iou > iou_threshold:
                is_riding = True
                break
        if not is_riding:
            refined_person_count += 1
    return refined_person_count

def process_image(uploaded_file, model):
    image = Image.open(uploaded_file)
    image = preprocess_image(image)
    results = model(image)
    
    # Count specific objects and remove persons riding motorcycles
    labels = results.names
    boxes = results.xyxy[0].cpu().numpy()
    
    counts = {"car": 0, "motorcycle": 0, "person": 0}
    person_boxes = []
    motorcycle_boxes = []
    car_boxes = []

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(boxes):
        label = labels[int(cls)]
        if label == "person":
            person_boxes.append((x1, y1, x2, y2))
        elif label == "motorcycle":
            motorcycle_boxes.append((x1, y1, x2, y2))
            counts['motorcycle'] += 1
        elif label == "car":
            car_boxes.append((x1, y1, x2, y2))
            counts['car'] += 1

    # Evaluate and refine person count across different IoU thresholds
    best_person_count = len(person_boxes)
    best_threshold = 0.0
    for _ in range(10):  # Repeat the evaluation 10 times
        for threshold in np.arange(0.1, 1.0, 0.01):  # Use a smaller step for finer evaluation
            refined_person_count = refine_person_count(person_boxes, motorcycle_boxes, threshold)
            if refined_person_count < best_person_count:
                best_person_count = refined_person_count
                best_threshold = threshold

    counts['person'] = best_person_count

    # Draw bounding boxes for motorcycles and cars without additional labels
    image_np = np.array(image)
    for (mx1, my1, mx2, my2) in motorcycle_boxes:
        image_np = cv2.rectangle(image_np, (int(mx1), int(my1)), (int(mx2), int(my2)), (0, 255, 0), 2)
        image_np = cv2.putText(image_np, "motorcycle", (int(mx1), int(my1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for (cx1, cy1, cx2, cy2) in car_boxes:
        image_np = cv2.rectangle(image_np, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)
        image_np = cv2.putText(image_np, "car", (int(cx1), int(cy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    result_image = Image.fromarray(image_np)

    # Logging for evaluation
    st.write(f"Best IoU threshold: {best_threshold}")
    st.write(f"Total persons detected: {len(person_boxes)}")
    st.write(f"Persons removed (riding motorcycles): {len(person_boxes) - best_person_count}")

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
