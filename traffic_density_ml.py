import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pip install --upgrade pip

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
    folder = 'path_to_images_folder'
    images = load_images_from_folder(folder)
    if not images:
        raise FileNotFoundError(f"No images found in folder: {folder}")

    X = preprocess_images(images)
    labels = pd.read_csv('path_to_labels.csv')
    y = labels['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    main()
