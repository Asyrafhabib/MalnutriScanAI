import os
import pandas as pd
from model import load_saved_models

def perbaiki_csv():
    print("Memuat model XGBoost dari model.py...")
    models_dict, le, scaler, numeric_cols = load_saved_models()

    if not models_dict or "XGBoost" not in models_dict:
        raise RuntimeError("Gagal memuat XGBoost model. Pastikan model.py berhasil memuat model dari folder model.")

    model = models_dict["XGBoost"]
    
    # Baca CSV yang statusnya mau dikoreksi
    file_path = 'data/prediction_history.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        print("CSV kosong, tidak ada data untuk dikoreksi.")
        return

    print(f"Mengoreksi {len(df)} baris data...")
    # Ambil kolom numerik sesuai urutan yang diminta oleh model.py
    X = df[numeric_cols]

    # Minta XGBoost memprediksi ulang semuanya
    prediksi = model.predict(X)
    
    # Ubah hasil angka menjadi teks menggunakan label encoder
    label_asli = le.inverse_transform(prediksi)
    df['nutrition_status'] = [label.lower() for label in label_asli]

    df.to_csv(file_path, index=False)
    print("Selesai! CSV sudah sinkron dengan hasil prediksi XGBoost.")

if __name__ == "__main__":
    perbaiki_csv()