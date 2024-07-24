import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    st.title("Traffic Density Analysis")
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

if __name__ == "__main__":
    main()
