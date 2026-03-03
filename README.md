# 👶 Malnutrition AI: Child Stunting Detection System

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Data Source](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Academic_Project-success)
![License](https://img.shields.io/badge/License-MIT-green)

> Sebuah sistem kecerdasan buatan berbasis Python untuk menganalisis data numerik dan mendeteksi risiko *stunting* (tengkes) pada anak sejak dini, lengkap dengan visualisasi data, laporan statistik, dan rekomendasi nutrisi.

## 📖 Latar Belakang
Malnutrisi dan *stunting* merupakan tantangan kesehatan global yang membutuhkan penanganan cepat dan tepat. Proyek **Malnutrition AI** ini dikembangkan untuk mengotomatisasi proses identifikasi status gizi anak berdasarkan parameter numerik (umur, berat badan, tinggi badan, jenis kelamin). 

Sistem ini membandingkan kinerja beberapa algoritma *Machine Learning* untuk menemukan model prediksi terbaik, memberikan wawasan visual, serta menyajikan saran tindak lanjut sebagai alat bantu pengambilan keputusan awal.

## ✨ Fitur Unggulan
- 📊 **Analisis Gizi Otomatis:** Mengklasifikasikan status gizi anak berdasarkan standar kesehatan menggunakan model *Machine Learning* yang telah dioptimasi.
- 📈 **Visualisasi Data komprehensif:** Menghasilkan grafik distribusi data (boxplot, bar chart) dan *confusion matrix* untuk evaluasi model.
- 📋 **Laporan Statistik & Evaluasi:** Menyediakan metrik evaluasi yang detail (*Accuracy*, *F1-Score*, *Classification Report*).
- 🍎 **Sistem Rekomendasi:** Memberikan panduan nutrisi dasar dan langkah intervensi berdasarkan hasil prediksi.

## 📂 Sumber Data (Dataset)
Model dalam proyek ini dilatih menggunakan dataset publik terverifikasi dari Kaggle:
🔗 **[Children Malnutrition Dataset by Albert Kingstone](https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset)**

## 🛠️ Teknologi & Library yang Digunakan
Proyek ini dibangun menggunakan ekosistem Python dengan pustaka (*library*) utama sebagai berikut:

* **Manipulasi & Analisis Data Numerik:**
  * `pandas`: Antarmuka untuk impor, manipulasi, dan analisis data tabular.
  * `numpy`: Antarmuka untuk operasi numerik dan perhitungan *array*.
* **Machine Learning Pipeline (`scikit-learn`):**
  * **Preprocessing:** `MinMaxScaler` (normalisasi data) & `LabelEncoder` (encoding label kategori).
  * **Data Splitting & Tuning:** `train_test_split` & `GridSearchCV` (optimasi *hyperparameter*).
  * **Algoritma Klasifikasi:** Menguji dan membandingkan `KNeighborsClassifier`, `SVC`, `DecisionTreeClassifier`, dan `RandomForestClassifier`.
  * **Evaluasi Model:** Menggunakan `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`, dan `ConfusionMatrixDisplay`.
* **Visualisasi Data:**
  * `matplotlib.pyplot`: Membuat grafik evaluasi, *boxplot*, dan *bar chart*.
* **Deployment & Penyimpanan:**
  * `joblib`: Digunakan untuk mengekspor dan menyimpan model terlatih agar siap digunakan pada tahap produksi (*deployment-ready*).