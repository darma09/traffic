import streamlit as st
import time
import threading

def traffic_light_simulation(north, south, east, west, stop_event):
    traffic = {'Utara': north, 'Selatan': south, 'Timur': east, 'Barat': west}
    sorted_traffic = sorted(traffic.items(), key=lambda x: x[1], reverse=True)
    
    cycle_times = {direction: max(3, vehicles // 10 * 3) for direction, vehicles in sorted_traffic}
    
    st.write("### Lampu Lalu Lintas Perempatan")
    st.write("Urutan lampu lalu lintas berdasarkan kepadatan kendaraan:")

    for direction, vehicles in sorted_traffic:
        st.write(f"{direction}: {vehicles} kendaraan, {cycle_times[direction]} detik lampu hijau")
    
    while not stop_event.is_set():
        for direction, _ in sorted_traffic:
            if stop_event.is_set():
                break
            st.write(f"#### Lampu hijau untuk arah {direction} selama {cycle_times[direction]} detik")
            with st.spinner(f"Lampu hijau di arah {direction}..."):
                for i in range(cycle_times[direction]):
                    if stop_event.is_set():
                        break
                    st.image('https://via.placeholder.com/100x100.png?text=Hijau', caption=f"Lampu hijau {direction}", use_column_width=True)
                    time.sleep(1)
                st.image('https://via.placeholder.com/100x100.png?text=Merah', caption=f"Lampu merah {direction}", use_column_width=True)
            time.sleep(1)

def main():
    st.title('Simulasi Lampu Lalu Lintas Perempatan')
    st.markdown("""
        Masukkan jumlah kendaraan di masing-masing arah untuk mensimulasikan skema lampu lalu lintas.
        Sistem akan mengatur durasi lampu hijau berdasarkan kepadatan kendaraan.
    """)

    north = st.number_input('Jumlah kendaraan dari Utara:', min_value=0, value=0)
    south = st.number_input('Jumlah kendaraan dari Selatan:', min_value=0, value=0)
    east = st.number_input('Jumlah kendaraan dari Timur:', min_value=0, value=0)
    west = st.number_input('Jumlah kendaraan dari Barat:', min_value=0, value=0)

    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if st.button('Mulai Simulasi'):
        st.session_state.stop_event.clear()
        threading.Thread(target=traffic_light_simulation, args=(north, south, east, west, st.session_state.stop_event)).start()

    if st.button('Hentikan Simulasi'):
        st.session_state.stop_event.set()

if __name__ == "__main__":
    main()
