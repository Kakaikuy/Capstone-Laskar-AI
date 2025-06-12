# Prediksi Kemacetan Lalu Lintas DKI Jakarta Menggunakan Artificial Neural Network (ANN)

Proyek ini bertujuan untuk membangun model prediksi kemacetan di wilayah DKI Jakarta menggunakan pendekatan Artificial Neural Network (ANN). Dengan memanfaatkan data wilayah, jumlah penduduk, dan panjang jalan, model ini diharapkan mampu memberikan estimasi tingkat kemacetan secara akurat.

## 📌 Deskripsi Proyek

Kemacetan menjadi masalah utama di Jakarta. Dengan pendekatan pemodelan berbasis data, proyek ini memanfaatkan data statistik dari tiap wilayah di DKI Jakarta, dan menggunakan model ANN untuk mengestimasi tingkat kemacetan sebagai output.

## 🎯 Tujuan

- Mengintegrasikan berbagai sumber data wilayah Jakarta
- Melatih model ANN untuk mengestimasi tingkat kemacetan
- Mengevaluasi performa model secara kuantitatif

## 🗃️ Dataset

Dataset utama:  
📄 `dataset_gabungan_dki.csv`  
Dataset ini merupakan hasil penggabungan beberapa sumber data:

- **Jumlah penduduk** per wilayah dari tahun 2020–2023
- **Panjang jalan** (km) per tahun dan wilayah
- **Luas wilayah dan kepadatan penduduk**
- **Ringkasan wilayah administratif DKI Jakarta**

Fitur yang digunakan:
- Tahun
- Kota/Kabupaten
- Luas wilayah
- Jumlah penduduk
- Panjang jalan
- Kepadatan penduduk
- Target: Estimasi Kemacetan (berbasis klasifikasi/estimasi model)

Semua data berada di dalam folder `Data/` dan `Dashboard/`.

## ⚙️ Teknologi & Library

- Python 3.x
- TensorFlow 2.15.0
- Scikit-learn 1.4.2
- Pandas 2.2.2
- NumPy 1.24.4
- Matplotlib 3.8.4
- Seaborn 0.13.2
- Streamlit 1.35.0

## 🧠 Arsitektur Model

Model ANN dibangun menggunakan `Sequential` API dari TensorFlow Keras:

- Dense (64 units)
- Dense (128 units)
- Dense (128 units)
- Output layer

Optimizer: Adam  
Loss function: Categorical/MAE (tergantung task)  
Epochs: 200  
Batch size: 16  
Validation split: 0.2

## 📈 Hasil Model

- **Akurasi** pada data validasi: **91.28%**
- Model berhasil menyimpan file `.h5` dan scaler `.pkl` untuk deployment

## 🚀 Cara Menjalankan Proyek

### 1. Install Dependencies

Gunakan virtual environment (opsional):

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install pustaka yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 2. Jalankan Dashboard Streamlit

Pindah ke folder dashboard (misal: `Dashboard/`):

```bash
cd Dashboard
streamlit run app.py
```

Jika file utama memiliki nama lain, sesuaikan:

```bash
streamlit run dashboard.py
```

Setelah itu, Streamlit akan membuka browser secara otomatis

### 4. Struktur File Penting

```
├── Dashboard/
│   ├── Dashboard.py              # File utama Streamlit
│   ├── model_kemacetan.h5        # Model ANN hasil pelatihan
│   ├── scaler.pkl                # File scaler untuk normalisasi data
│   └── dataset_gabungan_dki.csv  # Dataset hasil merge & cleaning
```

Pastikan file model `.h5`, scaler `.pkl`, dan dataset berada pada direktori yang sama atau telah diatur path-nya di dalam `app.py`.

## 📈 Hasil Model

- Akurasi data testing: **91.28%**
- Model ANN terdiri dari 3 hidden layer dengan 64–128 neuron
- Fitur input: `luas_km2`, `kepadatan`, `jumlah_penduduk`, `panjang_jalan`
- Target output: `kemacetan` (0 = tidak macet, 1 = macet)