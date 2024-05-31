import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv"
    data = pd.read_csv(url)
    return data

# Preprocess the data
def preprocess_data(data):
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek
    data['month'] = data['date_time'].dt.month
    
    features = ['hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
    X = data[features]
    y = data['traffic_volume']
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, X_test, y_test, y_pred

# Plot the results
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Volume')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Volume')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume')
    st.pyplot(plt)

# Main function for Streamlit
def main():
    st.title('Traffic Flow Density Prediction')

    data = load_data()
    st.write('### Data Preview')
    st.write(data.head())

    X, y = preprocess_data(data)
    model, mse, X_test, y_test, y_pred = train_model(X, y)

    st.write(f'### Mean Squared Error: {mse}')
    
    st.write('### Prediction Results')
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
