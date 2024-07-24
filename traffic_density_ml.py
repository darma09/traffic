import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure the 'ultralytics' package is installed
try:
    import ultralytics
except ImportError:
    os.system("pip install ultralytics")
    import ultralytics

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8x.pt')  # Using the more accurate model

# Function to preprocess image for YOLOv8
def preprocess_image(image):
    image = image.convert('RGB')
    image = np.array(image)
    return image

# Function to extract features from bounding boxes
def extract_features(image, boxes):
    features = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cropped_image = image[y_min:y_max, x_min:x_max]
        resized_image = cv2.resize(cropped_image, (64, 64))  # Resize for consistency
        flattened = resized_image.flatten()
        features.append(flattened)
    return np.array(features)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess and make predictions
    processed_image = preprocess_image(image)
    results = model(processed_image)

    # Initialize bounding boxes and labels
    boxes = []
    labels = []
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        boxes.append(bbox)
        labels.append(cls)

    # Extract features from bounding boxes
    features = extract_features(processed_image, boxes)

    # Dummy data for training Random Forest (replace with real annotated data)
    X = features  # Features extracted from bounding boxes
    y = labels    # Corresponding labels (0 for pedestrian, 1 for pedestrian on motorcycle)

    # Train a Random Forest classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

    # Use the trained Random Forest classifier to predict the labels of detected objects
    predictions = rf_classifier.predict(features)

    # Display results
    pedestrian_count = sum(1 for pred in predictions if pred == 'person')
    st.write(f"Pedestrians: {pedestrian_count}")

    # Draw bounding boxes on the image
    annotated_image = processed_image.copy()
    for bbox, label, pred in zip(boxes, labels, predictions):
        color = (0, 255, 0) if pred == 'person' else (255, 0, 0)
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label_text = f"{pred.capitalize()}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_ymin = max(bbox[1], label_size[1] + 10)
        cv2.rectangle(annotated_image, (bbox[0], label_ymin - label_size[1] - 10), (bbox[0] + label_size[0], label_ymin + 5), color, cv2.FILLED)
        cv2.putText(annotated_image, label_text, (bbox[0], label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the annotated image
    st.image(annotated_image, caption='Detected Image with Refined Predictions', use_column_width=True)
