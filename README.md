👶 MalnutriScan AI
Yapay Zeka Destekli Çocuk Malnütrisyon Risk Tahmin Sistemi

MalnutriScan AI, çocuklarda malnütrisyon ve stunting riskini erken dönemde tespit etmek amacıyla geliştirilmiş yapay zeka destekli bir karar destek sistemidir.

Sistem; yaş, cinsiyet, kilo, boy, MUAC (Mid-Upper Arm Circumference), iştah skoru ve klinik değerlendirme verilerini kullanarak XGBoost algoritması ile risk tahmini gerçekleştirmektedir.

Proje Özeti

Bu proje kapsamında çocukların antropometrik ölçümleri analiz edilerek beslenme durumları değerlendirilmekte ve olası malnütrisyon riskleri belirlenmektedir.

Model, Kaggle veri seti üzerinde eğitilmiş ve optimize edilmiş XGBoost algoritmasını kullanmaktadır.

Tahmin sonuçları modern bir web arayüzü üzerinden sunulmakta ve kullanıcıya detaylı analiz raporları sağlanmaktadır.

Özellikler
Dashboard
Model performans göstergeleri
Accuracy, Recall, Precision ve F1 Score sonuçları
Veri seti istatistikleri
Risk dağılım analizleri
AI Prediction
Yaş (Ay)
Cinsiyet
Kilo (kg)
Boy (cm)
MUAC ölçümü
Kilo kaybı bilgisi
İştah skoru

girdileri kullanılarak malnütrisyon riski tahmini yapılmaktadır.

Analytics Dashboard
Tahmin geçmişi
Risk dağılım grafikleri
Yaş grubu analizleri
Cinsiyet dağılımı
BMI analizleri
CSV dışa aktarma desteği
Kullanılan Teknolojiler
Frontend
Streamlit
Veri İşleme
Pandas
NumPy
Makine Öğrenmesi
XGBoost
Scikit-Learn
Veri Görselleştirme
Plotly
Matplotlib
Model Kaydetme
Joblib
Veri Seti

Children Malnutrition Dataset

Kaynak:

https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset

Model Performansı
Metrik	Sonuç
Accuracy	95.9%
Recall	95.9%
Precision	95.9%
F1 Score	95.9%

Kurulum
pip install -r requirements.txt

Çalıştırma
streamlit run app.py

MacOS kullanıcıları için:

brew install libomp

Proje Yapısı
MalnutriScanAI
│
├── app.py
├── analytics.py
├── dashboard.py
├── prediction.py
├── navigation.py
├── yapayzeka.py
│
├── model/
├── data/
│
├── requirements.txt
└── README.md
Geliştirici

Bu proje akademik amaçlarla geliştirilmiştir.
