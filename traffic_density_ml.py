import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
import io

# Function to load YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers
    except cv2.error as e:
        st.error("Failed to load YOLO model. Please ensure 'yolov4.weights' and 'yolov4.cfg' are available.")
        raise e

@st.cache_resource
def load_data_from_url(url):
    response = requests.get(url)
    drive_file = io.StringIO(response.content.decode('utf-8'))
    data = pd.read_csv(drive_file)
    return data

@st.cache_resource
def load_classes():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def process_image(uploaded_file, net, output_layers, classes):
    image = Image.open(uploaded_file)
    image = np.array(image)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

# Step 4: File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Processing image...")
    
    # Load YOLO model
    net, output_layers = load_yolo_model()

    # Load class labels
    classes = load_classes()

    # Step 5: Analyze the uploaded image
    result_image = process_image(uploaded_file, net, output_layers, classes)

    # Step 6: Display the result of image analysis
    st.image(result_image, caption='Processed Image', use_column_width=True)
else:
    st.write("Please upload an image file.")
