import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv"
    data = pd.read_csv(url)
    return data

# Pra-pemrosesan data
def preprocess_data(data):
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek
    data['month'] = data['date_time'].dt.month

    features = ['hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
    X = data[features]
    y = data['traffic_volume']
    return X, y

# Melatih model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test, y_test, y_pred

# Plot hasil
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Volume Aktual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Volume Prediksi')
    plt.legend()
    plt.xlabel('Indeks')
    plt.ylabel('Volume Lalu Lintas')
    plt.title('Volume Lalu Lintas Aktual vs Prediksi')
    st.pyplot(plt)

# Fungsi utama untuk Streamlit
def main():
    st.set_page_config(page_title='Prediksi Kepadatan Lalu Lintas', layout='wide')
    
    st.title('Prediksi Kepadatan Lalu Lintas')
    st.markdown("""
        Aplikasi ini menggunakan model pembelajaran mesin untuk memprediksi kepadatan lalu lintas berdasarkan berbagai faktor seperti waktu, cuaca, dan kondisi lainnya.
        Dataset yang digunakan untuk melatih model bersumber dari [Metro Interstate Traffic Volume](https://github.com/darma09/traffic).
    """)

    data = load_data()

    st.sidebar.header('Ikhtisar Data')
    st.sidebar.write('### Pratinjau Data')
    st.sidebar.dataframe(data.head())

    st.sidebar.write('### Ringkasan Data')
    st.sidebar.write(data.describe())

    X, y = preprocess_data(data)
    model, mse, r2, X_test, y_test, y_pred = train_model(X, y)

    st.write(f'### Kinerja Model')
    st.write(f'- Mean Squared Error: {mse}')
    st.write(f'- R-squared: {r2}')

    st.write('### Hasil Prediksi')
    plot_results(y_test, y_pred)

    st.write('### Koefisien Fitur')
    feature_importances = pd.DataFrame(model.coef_, X.columns, columns=['Koefisien'])
    st.bar_chart(feature_importances)

    st.write('### Pairplot Fitur')
    sns.pairplot(data[['traffic_volume', 'hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']])
    st.pyplot(plt)

if __name__ == "__main__":
    main()
