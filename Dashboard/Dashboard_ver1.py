import os
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'model_kemacetan.h5')
scaler_path = os.path.join(base_path, 'scaler.pkl')

try:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

st.title("Prediksi Kemacetan")
st.markdown("Masukkan fitur wilayah untuk memprediksi apakah wilayah tersebut **macet** (1) atau **tidak** (0).")
st.header("Input Data Wilayah")

luas = st.number_input("Luas Daerah (km²)", min_value=0.0, value=10.0, step=0.1)
kepadatan = st.number_input("Kepadatan (jiwa/km²)", min_value=0.0, value=5000.0, step=100.0)
penduduk = st.number_input("Jumlah Penduduk", min_value=0, value=100000, step=1000)
panjang_jln = st.number_input("Panjang Jalan (km)", min_value=0.0, value=500.0, step=1.0)

if st.button("Prediksi Kemacetan"):
    x = np.array([[luas, kepadatan, penduduk, panjang_jln]])
    x_scaled = scaler.transform(x)
    y_prob = model.predict(x_scaled)[0, 0]
    if y_prob < 0.4:
        kategori = "🚗 Lancar"
    elif y_prob < 0.7:
        kategori = "🚙 Padat"
    else:
        kategori = "🚦 Macet"
    st.subheader(f"Hasil Prediksi: {kategori}")
    st.write(f"Probabilitas kemacetan: {y_prob:.2%}")
