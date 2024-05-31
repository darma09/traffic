import streamlit as st
import time

def traffic_light_simulation(north, south, east, west):
    traffic = {'Utara': north, 'Selatan': south, 'Timur': east, 'Barat': west}
    sorted_traffic = sorted(traffic.items(), key=lambda x: x[1], reverse=True)
    
    cycle_times = {direction: max(3, vehicles // 10 * 3) for direction, vehicles in sorted_traffic}

    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'direction_idx' not in st.session_state:
        st.session_state.direction_idx = 0

    directions = [direction for direction, _ in sorted_traffic]
    current_direction = directions[st.session_state.direction_idx]
    current_cycle_time = cycle_times[current_direction]
    
    st.write("### Lampu Lalu Lintas Perempatan")
    st.write("Urutan lampu lalu lintas berdasarkan kepadatan kendaraan:")
    for direction, vehicles in sorted_traffic:
        st.write(f"{direction}: {vehicles} kendaraan, {cycle_times[direction]} detik lampu hijau")

    st.write(f"#### Lampu hijau untuk arah {current_direction} selama {current_cycle_time} detik")
    st.image('https://via.placeholder.com/100x100.png?text=Hijau', caption=f"Lampu hijau {current_direction}", use_column_width=True)
    
    if st.session_state.step < current_cycle_time:
        st.session_state.step += 1
    else:
        st.session_state.step = 0
        st.session_state.direction_idx = (st.session_state.direction_idx + 1) % len(directions)

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
        st.session_state.stop_simulation = False
        st.session_state.step = 0
        st.session_state.direction_idx = 0

    if st.button('Hentikan Simulasi'):
        st.session_state.stop_simulation = True

    if not st.session_state.get('stop_simulation', True):
        traffic_light_simulation(north, south, east, west)
        time.sleep(1)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
