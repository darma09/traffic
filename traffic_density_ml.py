import streamlit as st
import pandas as pd
import joblib
import gdown

@st.cache_data
def load_csv_data(url):
    try:
        data = pd.read_csv(url)
        st.write("CSV data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")
        raise e

@st.cache_data
def download_and_load_model(url, model_path):
    try:
        gdown.download(url, model_path, quiet=False)
        random_forest_model = joblib.load(model_path)
        st.write("Random Forest model loaded successfully.")
        return random_forest_model
    except Exception as e:
        st.error(f"Error loading the Random Forest model: {str(e)}")
        raise e

# Load CSV data
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
data = load_csv_data(csv_url)
st.dataframe(data)  # Display the loaded CSV file in Streamlit

# Install scikit-learn
try:
    import sklearn
    st.write(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn"])
    import sklearn
    st.write(f"scikit-learn version: {sklearn.__version__}")

# Download and load the Random Forest model from Google Drive
url = 'https://drive.google.com/uc?id=1l5PvNkp3Lq8O9U41UvuW4MfaMCLyo57W'
model_path = 'random_forest_model.pkl'
random_forest_model = download_and_load_model(url, model_path)

# Display information about the Random Forest model
st.write(random_forest_model)
