import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model dan scaler
model = tf.keras.models.load_model('model_kemacetan.h5')
scaler = joblib.load('scaler.pkl')

st.title("🚗 Prediksi Kemacetan Wilayah")
st.markdown("""
Masukkan fitur wilayah untuk memprediksi  
apakah wilayah tersebut **macet** (1) atau **tidak** (0).
""")

# Input fitur – otomatis update (tanpa Enter)
luas        = st.number_input("Luas Wilayah (km²)", min_value=0.0, step=0.1)
kepadatan   = st.number_input("Kepadatan (jiwa/km²)", min_value=0.0, step=1.0)
penduduk    = st.number_input("Jumlah Penduduk", min_value=0.0, step=1.0)
panjang_jln = st.number_input("Panjang Jalan (km)", min_value=0.0, step=0.1)

# Tombol Prediksi
if st.button("Prediksi Kemacetan"):
    if luas == 0 or kepadatan == 0 or penduduk == 0 or panjang_jln == 0:
        st.warning("Harap isi semua fitur dengan benar.")
    else:
        # Siapkan data & scaling
        x = np.array([[luas, kepadatan, penduduk, panjang_jln]])
        x_scaled = scaler.transform(x)

        # Prediksi probabilitas
        y_prob = model.predict(x_scaled)[0, 0]
        st.subheader(f"🔎 Probabilitas Kemacetan: {y_prob:.2%}")

        # Interpretasi probabilitas
        if y_prob < 0.4:
            status = "🚙 Lalu lintas **lancar**"
        elif y_prob < 0.7:
            status = "🚕 Lalu lintas **padat**"
        else:
            status = "🚗 Lalu lintas **macet**"

        st.success(status)

        # Label klasifikasi biner
        label = int(y_prob > 0.5)
        st.write(f"Prediksi klasifikasi (biner): {'Macet' if label else 'Tidak Macet'}")