import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
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
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test, y_test, y_pred

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
    st.set_page_config(page_title='Traffic Flow Density Prediction', layout='wide')
    
    st.title('Traffic Flow Density Prediction')
    st.markdown("""
        This application uses a machine learning model to predict traffic flow density based on various factors such as time, weather, and other conditions.
        The dataset used for training the model is sourced from [Metro Interstate Traffic Volume](https://github.com/darma09/traffic).
    """)

    data = load_data()

    st.sidebar.header('Data Overview')
    st.sidebar.write('### Data Preview')
    st.sidebar.dataframe(data.head())

    st.sidebar.write('### Data Summary')
    st.sidebar.write(data.describe())

    X, y = preprocess_data(data)
    model, mse, r2, X_test, y_test, y_pred = train_model(X, y)

    st.write(f'### Model Performance')
    st.write(f'- Mean Squared Error: {mse}')
    st.write(f'- R-squared: {r2}')

    st.write('### Prediction Results')
    plot_results(y_test, y_pred)

    st.write('### Feature Importances')
    feature_importances = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    st.bar_chart(feature_importances)

    st.write('### Pairplot of Features')
    sns.pairplot(data[['traffic_volume', 'hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']])
    st.pyplot(plt)

if __name__ == "__main__":
    main()
