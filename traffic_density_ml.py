import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

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

def main():
    st.title('Traffic Density Prediction')

    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    if uploaded_files:
        images = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            images.append(img_array)
            st.image(img, caption=uploaded_file.name)

        try:
            X = preprocess_images(images)
            st.write(f"Preprocessed images shape: {X.shape}")
        except Exception as e:
            st.error(f"Error preprocessing images: {e}")
            return

        try:
            labels = pd.read_csv('path_to_labels.csv')
            y = labels['label'].values
            st.write(f"Loaded {len(y)} labels.")
        except Exception as e:
            st.error(f"Error loading labels: {e}")
            return

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Error splitting dataset: {e}")
            return

        try:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model accuracy: {accuracy}")
        except Exception as e:
            st.error(f"Error training or predicting model: {e}")

if __name__ == "__main__":
    main()
