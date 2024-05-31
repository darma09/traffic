import streamlit as st
import time

def traffic_light_simulation(north, south, east, west):
    traffic = {'Utara': north, 'Selatan': south, 'Timur': east, 'Barat': west}
    sorted_traffic = sorted(traffic.items(), key=lambda x: x[1], reverse=True)
    
    cycle_times = {direction: max(3, vehicles // 10 * 3) for direction, vehicles in sorted_traffic}
    
    st.write("### Lampu Lalu Lintas Perempatan")
    st.write("Urutan lampu lalu lintas berdasarkan kepadatan kendaraan:")

    for direction, vehicles in sorted_traffic:
        st.write(f"{direction}: {vehicles} kendaraan, {cycle_times[direction]} detik lampu hijau")
    
    stop_simulation = st.session_state.get('stop_simulation', False)

    while not stop_simulation:
        for direction, _ in sorted_traffic:
            if st.session_state.get('stop_simulation', False):
                return
            st.write(f"#### Lampu hijau untuk arah {direction} selama {cycle_times[direction]} detik")
            for i in range(cycle_times[direction]):
                if st.session_state.get('stop_simulation', False):
                    return
                st.image('https://via.placeholder.com/100x100.png?text=Hijau', caption=f"Lampu hijau {direction}", use_column_width=True)
                time.sleep(1)
                st.experimental_rerun()
            st.image('https://via.placeholder.com/100x100.png?text=Merah', caption=f"Lampu merah {direction}", use_column_width=True)
            time.sleep(1)
            st.experimental_rerun()

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

    if 'stop_simulation' not in st.session_state:
        st.session_state.stop_simulation = False

    if st.button('Mulai Simulasi'):
        st.session_state.stop_simulation = False
        traffic_light_simulation(north, south, east, west)

    if st.button('Hentikan Simulasi'):
        st.session_state.stop_simulation = True
        st.write("Simulasi dihentikan.")

if __name__ == "__main__":
    main()
