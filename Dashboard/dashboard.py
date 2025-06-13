import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import h5py

st.title("🚗 Prediksi Kemacetan Wilayah")
st.markdown("""
Masukkan fitur wilayah untuk memprediksi  
apakah wilayah tersebut **macet** (1) atau **tidak** (0).
(Isi data dengan desimal menggunakan ',')
""")

try:
    with h5py.File('model_kemacetan.h5', 'r') as f:
        st.info("✅ File model valid dan dapat dibuka.")
    try:
        model = tf.keras.models.load_model('model_kemacetan.h5')
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

except Exception as e:
    st.error(f"❌ File model_kemacetan.h5 rusak atau tidak bisa dibuka: {e}")
    st.stop()

# Load scaler
try:
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"❌ Gagal memuat scaler: {e}")
    st.stop()

# Input fitur
luas        = st.number_input("Luas Wilayah (km²)", min_value=0.0, value=1.88, step=0.1, help="Masukkan luas wilayah dalam satuan kilometer persegi")
kepadatan   = st.number_input("Kepadatan (jiwa/km²)", min_value=0.0, value=18178.52, step=1.0, help="Masukkan jumlah penduduk per km²")
penduduk    = st.number_input("Jumlah Penduduk", min_value=0.0, value=2434511.0, step=1.0, help="Masukkan total jumlah penduduk wilayah")
panjang_jln = st.number_input("Panjang Jalan (km)", min_value=0.0, value=1182.0, step=0.1, help="Masukkan total panjang jalan di wilayah dalam kilometer")

# Tombol Prediksi
if st.button("Prediksi Kemacetan"):
    if luas == 0 or kepadatan == 0 or penduduk == 0 or panjang_jln == 0:
        st.warning("Harap isi semua fitur dengan benar.")
    else:
        # Siapkan data & scaling
        x = np.array([[luas, kepadatan, penduduk, panjang_jln]])
        try:
            x_scaled = scaler.transform(x)
        except Exception as e:
            st.error(f"❌ Gagal melakukan transformasi data: {e}")
            st.stop()

        # Prediksi probabilitas
        try:
            y_prob = model.predict(x_scaled)[0, 0]
            st.subheader(f"🔎 Probabilitas Kemacetan: {y_prob:.2%}")

            # Interpretasi probabilitas
            if y_prob < 0.4:
                status = "Lalu lintas **LANCAR 🟢**"
            elif y_prob < 0.7:
                status = "Lalu lintas **PADAT 🟡**"
            else:
                status = "Lalu lintas **MACET 🔴**"

            st.success(status)

            # Label klasifikasi biner
            label = int(y_prob > 0.5)
            st.subheader(f"Hasil Prediksi: {status}")
            st.write(f"Prediksi klasifikasi (biner): {'Macet' if label else 'Tidak Macet'}")

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
