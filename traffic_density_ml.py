import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
import numpy as np
import time
import cv2

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

    features = ['hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'clouds_all']
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

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    image = image.resize((416, 416))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Fungsi untuk deteksi objek
def detect_objects(image, model, class_names, confidence_threshold=0.5):
    # Preproses gambar
    img = preprocess_image(image)
    # Prediksi deteksi objek
    detections = model.predict(img)
    boxes, scores, classes, nums = detections[0], detections[1], detections[2], detections[3]
    
    results = []
    for i in range(nums[0]):
        if scores[0][i] >= confidence_threshold:
            box = boxes[0][i]
            class_id = int(classes[0][i])
            class_name = class_names[class_id]
            results.append((box, scores[0][i], class_name))
    return results

# Fungsi untuk menggambar kotak pada objek yang terdeteksi
def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    for box, score, class_name in results:
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = xmin * image.width, xmax * image.width, ymin * image.height, ymax * image.height
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=2)
        draw.text((left, top), f"{class_name} {score:.2f}", fill="red")
    return image

def main():
    st.title('Simulasi Lampu Lalu Lintas Perempatan dan Deteksi Objek')

    data = load_data()
    X, y = preprocess_data(data)
    model, mse, r2, X_test, y_test, y_pred = train_model(X, y)

    st.sidebar.header('Prediksi Berdasarkan Waktu dan Cuaca')
    hour = st.sidebar.slider('Jam', 0, 23, 12)
    day_of_week = st.sidebar.selectbox('Hari dalam Minggu', ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'])
    day_of_week_map = {'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 'Jumat': 4, 'Sabtu': 5, 'Minggu': 6}
    day_of_week = day_of_week_map[day_of_week]
    month = st.sidebar.selectbox('Bulan', ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'])
    month_map = {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12}
    month = month_map[month]
    temp = st.sidebar.number_input('Suhu (K)', value=288.0)
    rain_type = st.sidebar.selectbox('Jenis Hujan', ['Tidak Hujan', 'Gerimis', 'Sedang', 'Deras', 'Badai', 'Es'])
    rain_intensity_map = {'Tidak Hujan': 0, 'Gerimis': 1, 'Sedang': 2, 'Deras': 3, 'Badai': 4, 'Es': 5}
    rain_intensity = rain_intensity_map[rain_type]
    clouds_all = st.sidebar.selectbox('Persentase Awan', ['Cerah', 'Berawan Sedikit', 'Berawan Banyak', 'Mendung'])
    clouds_all_map = {'Cerah': 0, 'Berawan Sedikit': 25, 'Berawan Banyak': 50, 'Mendung': 75}
    clouds_all = clouds_all_map[clouds_all]

    features = pd.DataFrame([[hour, day_of_week, month, temp, rain_intensity, clouds_all]], 
                            columns=['hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'clouds_all'])
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
    if st.button('Hentikan Simulasi'):
        st.session_state.stop_simulation = True

    if not st.session_state.get('stop_simulation', True):
        predictions = {'Utara': north, 'Selatan': south, 'Timur': east, 'Barat': west}
        traffic_light_simulation(predictions, rain_intensity)
        time.sleep(1)
        st.experimental_rerun()

    st.title('Deteksi Kendaraan dan Pejalan Kaki')
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)
        
        st.write("")
        st.write("Mengklasifikasi gambar...")
        
        # Muat model YOLO
        yolo_model = tf.saved_model.load("yolo_model_path")  # Ganti dengan path model YOLO Anda
        class_names = ["Mobil", "Motor", "Bis", "Pejalan Kaki"]  # Sesuaikan dengan class names Anda
        
        # Deteksi objek
        results = detect_objects(image, yolo_model, class_names)
        st.write(f"Hasil Deteksi: {results}")
        
        # Gambar kotak di sekitar objek yang terdeteksi
        image_with_boxes = draw_boxes(image, results)
        st.image(image_with_boxes, caption='Hasil Deteksi', use_column_width=True)

if __name__ == "__main__":
    main()
