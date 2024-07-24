import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import cv2
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib

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

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model_cnn = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    features = model_cnn.predict(image)
    return features.flatten()

# Load the pre-trained Random Forest model
random_forest_model = joblib.load('random_forest_model.pkl')

# Example function to classify objects using Random Forest
def classify_object(features):
    return random_forest_model.predict([features])[0]

# Function to calculate IoU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)

    # Calculate union
    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area
    return iou

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess and make predictions
    processed_image = preprocess_image(image)
    results = model(processed_image)

    # Initialize counters and bounding boxes
    car_count = 0
    motorcycle_count = 0
    pedestrian_count = 0
    person_boxes = []
    motorcycle_boxes = []

    # Collect bounding boxes and their confidences
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        bbox = box.xyxy[0].cpu().numpy()
        confidence = box.conf.item() * 100  # Get confidence as a percentage

        # Extract features using CNN
        cropped_img = processed_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        features = extract_features(cropped_img)

        # Classify object using Random Forest
        rf_cls = classify_object(features)

        if rf_cls == 'car':
            car_count += 1
        elif rf_cls == 'motorcycle':
            motorcycle_count += 1
            motorcycle_boxes.append(bbox)
        elif rf_cls == 'person':
            person_boxes.append(bbox)

    # Count pedestrians not on motorcycles
    for person_box in person_boxes:
        on_motorcycle = False
        for motorcycle_box in motorcycle_boxes:
            iou = calculate_iou(person_box, motorcycle_box)
            if iou > 0.5:  # Consider as on motorcycle if IoU > 0.5
                on_motorcycle = True
                break
        if not on_motorcycle:
            pedestrian_count += 1

    # Display the results
    st.write(f"Cars: {car_count}")
    st.write(f"Motorcycles: {motorcycle_count}")
    st.write(f"Pedestrians: {pedestrian_count}")

    # Draw bounding boxes and confidences on the image
    annotated_image = processed_image.copy()
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        confidence = box.conf.item() * 100  # Get confidence as a percentage
        label = f"{cls.capitalize()}: {confidence:.2f}%"
        
        # Draw bounding box
        color = (0, 255, 0) if cls == 'person' else (255, 0, 0) if cls == 'motorcycle' else (0, 0, 255)
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_ymin = max(bbox[1], label_size[1] + 10)
        cv2.rectangle(annotated_image, (bbox[0], label_ymin - label_size[1] - 10), (bbox[0] + label_size[0], label_ymin + 5), color, cv2.FILLED)
        cv2.putText(annotated_image, label, (bbox[0], label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the annotated image
    st.image(annotated_image, caption='Detected Image with Confidences', use_column_width=True)
