import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

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

# Fungsi untuk menentukan waktu lampu hijau berdasarkan kepadatan lalu lintas
def calculate_cycle_time(volume, rain_intensity):
    base_time = max(3, min(volume // 10 * 3, 60))
    rain_factor = 1 + rain_intensity * 0.2
    return int(base_time * rain_factor)

# Simulasi lampu lalu lintas
def traffic_light_simulation(predictions, rain_intensity):
    sorted_traffic = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    cycle_times = {direction: calculate_cycle_time(vehicles, rain_intensity) for direction, vehicles in sorted_traffic}

    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'direction_idx' not in st.session_state:
        st.session_state.direction_idx = 0

    directions = [direction for direction, _ in sorted_traffic]
    current_direction = directions[st.session_state.direction_idx]
    current_cycle_time = cycle_times[current_direction]
    remaining_time = current_cycle_time - st.session_state.step
    
    st.write("### Lampu Lalu Lintas Perempatan")
    st.write("Urutan lampu lalu lintas berdasarkan prediksi kepadatan kendaraan:")
    for direction, vehicles in sorted_traffic:
        st.write(f"{direction}: {int(vehicles)} kendaraan, {cycle_times[direction]} detik lampu hijau")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("Utara")
        st.image('https://via.placeholder.com/100x100.png?text=Hijau' if current_direction == 'Utara' else 'https://via.placeholder.com/100x100.png?text=Merah', use_column_width=True)
        if current_direction == 'Utara':
            st.write(f"Sisa waktu: {remaining_time} detik")
    with col2:
        st.write("Selatan")
        st.image('https://via.placeholder.com/100x100.png?text=Hijau' if current_direction == 'Selatan' else 'https://via.placeholder.com/100x100.png?text=Merah', use_column_width=True)
        if current_direction == 'Selatan':
            st.write(f"Sisa waktu: {remaining_time} detik")
    with col3:
        st.write("Timur")
        st.image('https://via.placeholder.com/100x100.png?text=Hijau' if current_direction == 'Timur' else 'https://via.placeholder.com/100x100.png?text=Merah', use_column_width=True)
        if current_direction == 'Timur':
            st.write(f"Sisa waktu: {remaining_time} detik")
    with col4:
        st.write("Barat")
        st.image('https://via.placeholder.com/100x100.png?text=Hijau' if current_direction == 'Barat' else 'https://via.placeholder.com/100x100.png?text=Merah', use_column_width=True)
        if current_direction == 'Barat':
            st.write(f"Sisa waktu: {remaining_time} detik")
    
    if st.session_state.step < current_cycle_time:
        st.session_state.step += 1
    else:
        st.session_state.step = 0
        st.session_state.direction_idx = (st.session_state.direction_idx + 1) % len(directions)

def main():
    st.title('Simulasi Lampu Lalu Lintas Perempatan')
    st.markdown("""
        Masukkan jumlah kendaraan di masing-masing arah untuk mensimulasikan skema lampu lalu lintas.
        Sistem akan mengatur durasi lampu hijau berdasarkan kepadatan kendaraan dan intensitas hujan.
    """)

    data = load_data()
    X, y = preprocess_data(data)
    model, mse, r2, X_test, y_test, y_pred = train_model(X, y)

    st.sidebar.header('Prediksi Berdasarkan Waktu dan Cuaca')
    hour = st.sidebar.slider('Jam', 0, 23, 12)
    day_of_week = st.sidebar.slider('Hari dalam Minggu', 0, 6, 0)
    month = st.sidebar.slider('Bulan', 1, 12, 1)
    temp = st.sidebar.number_input('Suhu', value=288.0)
    rain_type = st.sidebar.selectbox('Jenis Hujan', ['Tidak Hujan', 'Gerimis', 'Sedang', 'Deras', 'Badai', 'Es'])
    rain_intensity_map = {'Tidak Hujan': 0, 'Gerimis': 1, 'Sedang': 2, 'Deras': 3, 'Badai': 4, 'Es': 5}
    rain_intensity = rain_intensity_map[rain_type]
    snow_1h = st.sidebar.number_input('Curah Salju 1 Jam', value=0.0)
    clouds_all = st.sidebar.number_input('Persentase Awan', value=40.0)

    features = pd.DataFrame([[hour, day_of_week, month, temp, rain_intensity, snow_1h, clouds_all]], 
                            columns=['hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'snow_1h', 'clouds_all'])
    predicted_volume = model.predict(features)[0]

    north = st.number_input('Jumlah kendaraan dari Utara:', min_value=0, value=int(predicted_volume))
    south = st.number_input('Jumlah kendaraan dari Selatan:', min_value=0, value=int(predicted_volume))
    east = st.number_input('Jumlah kendaraan dari Timur:', min_value=0, value=int(predicted_volume))
    west = st.number_input('Jumlah kendaraan dari Barat:', min_value=0, value=int(predicted_volume))

    if 'stop_simulation' not in st.session_state:
        st.session_state.stop_simulation = False

    if st.button('Mulai Simulasi'):
        st.session_state.stop_simulation = False
        st.session_state.step = 0
        st.session_state.direction_idx = 0

    if st.button('Hentikan Simulasi'):
        st.session_state.stop_simulation = True

    if not st.session_state.get('stop_simulation', True):
        predictions = {'Utara': north, 'Selatan': south, 'Timur': east, 'Barat': west}
        traffic_light_simulation(predictions, rain_intensity)
        time.sleep(1)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
