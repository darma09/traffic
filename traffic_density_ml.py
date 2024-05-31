import streamlit as st
import time

def traffic_light_simulation(north, south, east, west):
    traffic = {'North': north, 'South': south, 'East': east, 'West': west}
    sorted_traffic = sorted(traffic.items(), key=lambda x: x[1], reverse=True)
    
    cycle_times = {direction: max(3, vehicles // 10 * 3) for direction, vehicles in sorted_traffic}
    
    st.write("### Lampu Lalu Lintas Perempatan")
    st.write("Urutan lampu lalu lintas berdasarkan kepadatan kendaraan:")

    for direction, vehicles in sorted_traffic:
        st.write(f"{direction}: {vehicles} kendaraan, {cycle_times[direction]} detik lampu hijau")
    
    for direction, _ in sorted_traffic:
        with st.spinner(f"Lampu hijau di arah {direction}..."):
            st.write(f"#### Lampu hijau untuk arah {direction} selama {cycle_times[direction]} detik")
            time.sleep(cycle_times[direction])

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

    if st.button('Mulai Simulasi'):
        traffic_light_simulation(north, south, east, west)

if __name__ == "__main__":
    main()
