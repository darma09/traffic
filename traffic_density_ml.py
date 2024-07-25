import streamlit as st
import pandas as pd
import joblib
import gdown
import numpy as np
import os

# Function to load CSV data
@st.cache_data
def load_csv_data(url):
    try:
        data = pd.read_csv(url)
        st.write("CSV data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")
        raise e

# Function to download and load the model
@st.cache_data
def download_and_modify_model(url, model_path):
    try:
        gdown.download(url, model_path, quiet=False)
        
        # Load the model
        model = joblib.load(model_path)
        
        # Modify the dtype of the node array
        for estimator in model.estimators_:
            tree = estimator.tree_
            tree_nodes = tree.__getstate__()['nodes']
            expected_dtype = np.dtype([
                ('left_child', '<i8'),
                ('right_child', '<i8'),
                ('feature', '<i8'),
                ('threshold', '<f8'),
                ('impurity', '<f8'),
                ('n_node_samples', '<i8'),
                ('weighted_n_node_samples', '<f8'),
                ('missing_go_to_left', 'u1')
            ])

            modified_nodes = np.zeros(tree_nodes.shape, dtype=expected_dtype)
            for field in tree_nodes.dtype.names:
                if field in expected_dtype.names:
                    modified_nodes[field] = tree_nodes[field]

            tree.__setstate__({'nodes': modified_nodes, **tree.__getstate__()})
        
        # Save the modified model
        modified_model_path = 'modified_random_forest_model.pkl'
        joblib.dump(model, modified_model_path)
        
        st.write("Random Forest model loaded and modified successfully.")
        return modified_model_path
    except Exception as e:
        st.error(f"Error loading the Random Forest model: {str(e)}")
        raise e

# Load CSV data
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
data = load_csv_data(csv_url)
st.dataframe(data)  # Display the loaded CSV file in Streamlit

# Ensure scikit-learn is installed
try:
    import sklearn
    st.write(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn==1.5.1"])
    import sklearn
    st.write(f"scikit-learn version: {sklearn.__version__}")

# Download and load the Random Forest model from Google Drive
url = 'https://drive.google.com/uc?id=1l5PvNkp3Lq8O9U41UvuW4MfaMCLyo57W'
model_path = 'random_forest_model.pkl'
modified_model_path = download_and_modify_model(url, model_path)

# Load the modified model
random_forest_model = joblib.load(modified_model_path)

# Display information about the Random Forest model
st.write(random_forest_model)
