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
    
    # Count specific objects and adjust bounding boxes
    labels = results.names
    counts = {"car": 0, "motorcycle": 0, "person": 0}
    motorcycle_boxes = []
    person_boxes = []
    adjusted_boxes = []

    for pred in results.pred[0]:
        label = labels[int(pred[-1])]
        bbox = pred[:4].tolist()
        if label == "motorcycle":
            motorcycle_boxes.append(bbox)
        elif label == "person":
            person_boxes.append(bbox)
        if label in counts:
            counts[label] += 1

    # Adjust the counting: check if a person is on a motorcycle
    for person_box in person_boxes:
        person_center = [(person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2]
        on_motorcycle = False
        for motorcycle_box in motorcycle_boxes:
            if (motorcycle_box[0] <= person_center[0] <= motorcycle_box[2]) and (motorcycle_box[1] <= person_center[1] <= motorcycle_box[3]):
                on_motorcycle = True
                break
        if on_motorcycle:
            counts["person"] -= 1
        else:
            adjusted_boxes.append((*person_box, 'person'))

    for motorcycle_box in motorcycle_boxes:
        adjusted_boxes.append((*motorcycle_box, 'motorcycle'))

    # Draw adjusted bounding boxes on the image
    result_image = np.array(result_image)
    for box in adjusted_boxes:
        x1, y1, x2, y2, label = box
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(result_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    result_image = Image.fromarray(result_image)
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
