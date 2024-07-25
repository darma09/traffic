import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Load CSV data
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
try:
    data = pd.read_csv(csv_url)
    st.write("CSV data loaded successfully.")
    st.dataframe(data)  # Display the loaded CSV file in Streamlit
except Exception as e:
    st.error(f"Error loading CSV data: {str(e)}")
    raise e

# Download the Random Forest model from Google Drive
url = 'https://drive.google.com/uc?id=1l5PvNkp3Lq8O9U41UvuW4MfaMCLyo57W'
model_path = 'random_forest_model.pkl'
try:
    gdown.download(url, model_path, quiet=False)
    # Ensure scikit-learn compatibility
    os.system("pip install scikit-learn==1.0.2")
    random_forest_model = joblib.load(model_path)
    st.write("Random Forest model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the Random Forest model: {str(e)}")
    raise e

# Display information about the Random Forest model
st.write(random_forest_model)
