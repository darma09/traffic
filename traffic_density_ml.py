import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images):
    processed_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        processed_images.append(resized.flatten())
    return np.array(processed_images)

def preprocess_image_for_cnn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_cars(model, img):
    preprocessed_img = preprocess_image_for_cnn(img)
    prediction = model.predict(preprocessed_img)
    return int(prediction[0] > 0.5)

def main():
    st.title("Traffic Density Analysis")

    # Image folder processing
    folder = st.text_input('Enter the path to the images folder:', 'path_to_images_folder')
    
    if st.button('Load and Process Images'):
        try:
            images = load_images_from_folder(folder)
            if not images:
                st.error(f"No images found in folder: {folder}")
                return
            
            X = preprocess_images(images)
            labels_path = st.text_input('Enter the path to the labels CSV file:', 'path_to_labels.csv')
            
            try:
                labels = pd.read_csv(labels_path)
                y = labels['label'].values
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"Model accuracy: {accuracy}")
                
            except FileNotFoundError:
                st.error(f"Labels file not found: {labels_path}")
            except Exception as e:
                st.error(f"An error occurred while processing the labels: {e}")
                
        except FileNotFoundError:
            st.error(f"Images folder not found: {folder}")
        except Exception as e:
            st.error(f"An error occurred while loading images: {e}")

    # Vehicle recognition app
    st.title("Vehicle Recognition App")
    st.write("Upload an image to identify and count cars.")
    
    model_path = st.text_input('Enter the path to the model file:', 'traffic_cnn_model.h5')
    
    if st.button('Load Model'):
        try:
            model = load_model(model_path)
            st.session_state['model'] = model  # Simpan model ke session state
            st.success(f"Model loaded successfully from {model_path}")
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

        if 'model' in st.session_state:
            num_cars = predict_cars(st.session_state['model'], img)  # Gunakan model dari session state
            st.write(f"Number of cars in the image: {num_cars}")
        else:
            st.error("Please load the model first.")

if __name__ == "__main__":
    main()
