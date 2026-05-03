import os
import pickle
import joblib
import streamlit as st
import xgboost as xgb
import pandas as pd

@st.cache_resource
def load_saved_models():
    """Load the latest XGBoost model and supporting artifacts from the 'model' folder."""
    model_dir = "model"
    pkl_path = os.path.join(model_dir, "xgboost_model.pkl")
    json_path = os.path.join(model_dir, "xgboost_model.json")
    label_path = os.path.join(model_dir, "label_encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    try:
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                xgb_model = pickle.load(f)
        elif os.path.exists(json_path):
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(json_path)
        else:
            raise FileNotFoundError("Tidak ada file XGBoost model (.pkl atau .json) di folder 'model'.")

        label_encoder = joblib.load(label_path)
        scaler = joblib.load(scaler_path)

        numeric_cols = ["age_months", "weight_kg", "height_cm", "muac_cm", "bmi"]

        return {"XGBoost": xgb_model}, label_encoder, scaler, numeric_cols

    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e}")
        st.info("Pastikan folder 'model' berisi xgboost_model.pkl atau xgboost_model.json, serta label_encoder.pkl dan scaler.pkl.")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None


if __name__ == "__main__":
    import warnings
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("⚙️  MEMUAT XGBoost MODEL TERBARU DARI FOLDER 'model/'...")
    print("=" * 60)

    models_dict, label_enc, scaler_obj, cols = load_saved_models()
    if not models_dict:
        print("❌ Gagal memuat model. Periksa file model dan artifact pendukung.")
        raise SystemExit(1)

    print(f"✅ Model yang dimuat: {list(models_dict.keys())}")
    print(f"✅ Tipe model: {type(models_dict['XGBoost']).__name__}")

    dataset_filename = "malnutrition_data .csv"
    if not os.path.exists(dataset_filename):
        dataset_filename = "malnutrition_data.csv"

    if not os.path.exists(dataset_filename):
        print(f"❌ Dataset tidak ditemukan: {dataset_filename}")
        raise SystemExit(1)

    df = pd.read_csv(dataset_filename).drop_duplicates()
    X = df[cols]
    y_true = label_enc.transform(df["nutrition_status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=42, stratify=y_true
    )

    y_pred = models_dict["XGBoost"].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Akurasi XGBoost pada data uji: {acc:.4f} ({acc*100:.1f}%)")
