import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import h5py
import os
import requests

st.title("🚗 Prediksi Kemacetan Wilayah")
st.markdown("""
Masukkan fitur wilayah untuk memprediksi  
apakah wilayah tersebut **macet** (1) atau **tidak** (0).  
(Isi data dengan desimal menggunakan ',')
""")

# ==== FUNGSI: Download dari Google Drive ====
@st.cache_resource(show_spinner=True)
def download_from_drive(file_id, filename):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Gagal mengunduh {filename} dari Google Drive.")
            st.stop()
    return filename

# ==== File ID Google Drive ====
MODEL_FILE_ID = "1Tyw7lpHjtb-89_PWG3o8WlxK0LG6nAuz"
SCALER_FILE_ID = "159U3pSEG4tYb3sk0i0GKyJiQHZs2WeEh"

# ==== Unduh dan load model ====
model_path = download_from_drive(MODEL_FILE_ID, "model_kemacetan.h5")
scaler_path = download_from_drive(SCALER_FILE_ID, "scaler.pkl")

try:
    with h5py.File(model_path, 'r') as f:
        st.info("✅ File model valid dan dapat dibuka.")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
except Exception as e:
    st.error(f"❌ File model rusak atau tidak bisa dibuka: {e}")
    st.stop()

# ==== Load scaler ====
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"❌ Gagal memuat scaler: {e}")
    st.stop()

# ==== Input fitur ====
luas = st.number_input("Luas Wilayah (km²)", min_value=0.0, value=1.88, step=0.1)
kepadatan = st.number_input("Kepadatan (jiwa/km²)", min_value=0.0, value=18178.52, step=1.0)
penduduk = st.number_input("Jumlah Penduduk", min_value=0.0, value=2434511.0, step=1.0)
panjang_jln = st.number_input("Panjang Jalan (km)", min_value=0.0, value=1182.0, step=0.1)

# ==== Prediksi ====
if st.button("Prediksi Kemacetan"):
    if luas == 0 or kepadatan == 0 or penduduk == 0 or panjang_jln == 0:
        st.warning("Harap isi semua fitur dengan benar.")
    else:
        try:
            x = np.array([[luas, kepadatan, penduduk, panjang_jln]])
            x_scaled = scaler.transform(x)
            y_prob = model.predict(x_scaled)[0, 0]

            st.subheader(f"🔎 Probabilitas Kemacetan: {y_prob:.2%}")
            if y_prob < 0.4:
                status = "Lalu lintas **LANCAR 🟢**"
            elif y_prob < 0.7:
                status = "Lalu lintas **PADAT 🟡**"
            else:
                status = "Lalu lintas **MACET 🔴**"

            st.success(status)
            label = int(y_prob > 0.5)
            st.subheader(f"Hasil Prediksi: {status}")
            st.write(f"Prediksi klasifikasi (biner): {'Macet' if label else 'Tidak Macet'}")

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
