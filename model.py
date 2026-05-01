import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


@st.cache_resource
def load_and_train_model():
    # ========================
    # DATA LOADING & PREPROCESSING
    # ========================
    df = pd.read_csv("malnutrition_data .csv")
    df = df.drop_duplicates()
    numeric_cols = ["age_months", "weight_kg", "height_cm", "muac_cm", "bmi"]
    
    # Label encoding
    le = LabelEncoder()
    df["nutrition_status_encoded"] = le.fit_transform(df["nutrition_status"])
    X = df[numeric_cols]
    y = df["nutrition_status_encoded"]
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE untuk balance data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Scaler untuk model yang memerlukan normalisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    
    # ========================
    # MODEL 1: LOGISTIC REGRESSION (Tuned)
    # ========================
    param_grid_log = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "class_weight": [None, "balanced"]
    }
    grid_log = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=42),
        param_grid=param_grid_log,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    grid_log.fit(X_train_scaled, y_train_smote)
    models["LogisticRegression"] = grid_log.best_estimator_
    
    # ========================
    # MODEL 2: RANDOM FOREST (Tuned)
    # ========================
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    random_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_distributions=param_grid_rf,
        n_iter=10,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42
    )
    random_rf.fit(X_train_smote, y_train_smote)
    models["RandomForest"] = random_rf.best_estimator_
    
    # ========================
    # MODEL 3: XGBOOST (DEFAULT - USED IN PREDICTION)
    # ========================
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model.fit(X_train_smote, y_train_smote)
    models["XGBoost"] = xgb_model
    
    # ========================
    # MODEL 4: CATBOOST
    # ========================
    cat_model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_state=42,
        verbose=0
    )
    cat_model.fit(X_train_smote, y_train_smote)
    models["CatBoost"] = cat_model
    
    # ========================
    # MODEL 5: SUPPORT VECTOR MACHINE (SVM)
    # ========================
    svm_model = SVC(kernel="rbf", random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    models["SVM"] = svm_model
    
    # ========================
    # MODEL 6: K-NEAREST NEIGHBORS (KNN)
    # ========================
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    models["KNN"] = knn_model
    
    # ========================
    # MODEL 7: NAIVE BAYES
    # ========================x
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    models["NaiveBayes"] = nb_model
    
    # ========================
    # MODEL 8: DECISION TREE
    # ========================
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    models["DecisionTree"] = dt_model
    
    return models, scaler, le, df, numeric_cols
