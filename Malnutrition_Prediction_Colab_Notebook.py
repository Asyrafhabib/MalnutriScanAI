# %% [markdown]
# # Malnutrition Prediction (Classification) — Google Colab Notebook
# This notebook follows a **step-by-step workflow similar to your heart-disease example**: import → explore → clean/noise check → split → normalize → train & tune models → evaluate → conclude.
# 
# **Dataset path (already mounted here):** `/mnt/data/malnutrition_data .csv`

# %% [markdown]
# 

# %%
# 1) Import Libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional: show more columns when printing
pd.set_option("display.max_columns", 200)


# %%
# 2) Import Dataset
DATA_PATH = "malnutrition_data.csv"  # <- change only if your file path differs in Colab

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head(10)


# %% [markdown]
# ## 3) Data Preprocessing
# ### 3.1 Dataset overview (`describe`-style summary)

# %%
def describe_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    variables, dtypes, count, unique, missing, min_, max_ = [], [], [], [], [], [], []
    for col in dataframe.columns:
        variables.append(col)
        dtypes.append(dataframe[col].dtype)
        count.append(len(dataframe[col]))
        unique.append(dataframe[col].nunique())
        missing.append(int(dataframe[col].isna().sum()))
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            min_.append(float(dataframe[col].min()))
            max_.append(float(dataframe[col].max()))
        else:
            min_.append("Str")
            max_.append("Str")
    return pd.DataFrame({
        "variable": variables,
        "dtype": dtypes,
        "count": count,
        "unique": unique,
        "missing_value": missing,
        "Min": min_,
        "Max": max_
    })

desc = describe_table(df)
desc


# %% [markdown]
# ### 3.2 Target & feature columns
# Your dataset has these columns:
# - **Features:** `age_months, weight_kg, height_cm, muac_cm, bmi`
# - **Target:** `nutrition_status` (example values: `normal`, `moderate`, `severe`)

# %%
# Auto-detect target (works even if you rename columns later)
possible_targets = ["nutrition_status", "target", "output", "label", "class", "y"]
target_col = next((c for c in possible_targets if c in df.columns), df.columns[-1])

feature_cols = [c for c in df.columns if c != target_col]
print("Target column:", target_col)
print("Feature columns:", feature_cols)

df[target_col].value_counts()


# %% [markdown]
# ### 3.3 Missing values + quick fixes

# %%
# Missing values check
missing = df.isna().sum().sort_values(ascending=False)
missing[missing > 0]


# %%
# If there are missing numeric values, fill with median (safe default).
# If you don't have missing values, this cell won't change anything.
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# If target is missing, drop those rows
df = df.dropna(subset=[target_col]).reset_index(drop=True)

print("Shape after missing-handling:", df.shape)


# %% [markdown]
# ### 3.4 Noise / outlier detection with boxplots (similar to your example)

# %%
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

for c in num_cols:
    plt.figure(figsize=(10, 2.5), dpi=140)
    plt.boxplot(df[c].dropna(), vert=False)
    plt.title(f"Boxplot — {c}")
    plt.xlabel(c)
    plt.show()


# %% [markdown]
# ### 3.5 Optional: Drop extreme outliers using IQR rule
# If your model becomes unstable, you can remove extreme outliers. This is optional — try **without** dropping first.

# %%
def drop_iqr_outliers(dataframe: pd.DataFrame, cols, factor: float = 3.0) -> pd.DataFrame:
    # factor=1.5 is strict; 3.0 is more conservative
    clean = dataframe.copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(clean[c]):
            continue
        q1 = clean[c].quantile(0.25)
        q3 = clean[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        clean = clean[(clean[c] >= lo) & (clean[c] <= hi)]
    return clean.reset_index(drop=True)

# Uncomment if you want to apply outlier removal:
# df = drop_iqr_outliers(df, num_cols, factor=3.0)
# print("Shape after outlier removal:", df.shape)


# %% [markdown]
# ### 3.6 Encode target + split dataset
# Because the target is text (e.g., `normal/moderate/severe`), we encode it to integers.

# %%
# Encode target labels (multiclass)
le = LabelEncoder()
y = le.fit_transform(df[target_col])

X = df[feature_cols].copy()

print("Classes mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
print("X shape:", X.shape, "| y shape:", y.shape)

# Train/test split (stratified keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40, stratify=y
)


# %% [markdown]
# ### 3.7 Normalize features (MinMaxScaler)

# %%
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Quick check
X_train_scaled[:3]


# %% [markdown]
# ## 4) Machine Learning Models + Hyperparameter Tuning
# We tune models using **GridSearchCV** (cleaner than manual loops, but same idea: try many settings and choose the best).

# %%
def evaluate_model(name, model, X_te, y_te):
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average="macro")
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1m, 4))
    print("\nClassification report:")
    print(classification_report(y_te, pred, target_names=le.classes_))

    cm = confusion_matrix(y_te, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix — {name}")
    plt.show()
    return acc, f1m


# %% [markdown]
# ### 4.1 KNN (tune `n_neighbors`, `p`, `weights`)

# %%
knn = KNeighborsClassifier()

param_grid_knn = {
    "n_neighbors": list(range(3, 31, 2)),
    "p": [1, 2],                 # Manhattan vs Euclidean
    "weights": ["uniform", "distance"]
}

gs_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring="f1_macro", n_jobs=-1)
gs_knn.fit(X_train_scaled, y_train)

print("Best KNN params:", gs_knn.best_params_)
best_knn = gs_knn.best_estimator_

acc_knn, f1_knn = evaluate_model("KNN", best_knn, X_test_scaled, y_test)


# %% [markdown]
# ### 4.2 SVM (tune `C`, `kernel`, `gamma`)

# %%
svm = SVC()

param_grid_svm = [
    {"kernel": ["linear"], "C": [0.1, 0.5, 1, 2, 5, 10]},
    {"kernel": ["rbf"], "C": [0.1, 0.5, 1, 2, 5, 10], "gamma": ["scale", "auto"]},
    {"kernel": ["poly"], "C": [0.1, 0.5, 1, 2, 5], "degree": [2, 3]}
]

gs_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring="f1_macro", n_jobs=-1)
gs_svm.fit(X_train_scaled, y_train)

print("Best SVM params:", gs_svm.best_params_)
best_svm = gs_svm.best_estimator_

acc_svm, f1_svm = evaluate_model("SVM", best_svm, X_test_scaled, y_test)


# %% [markdown]
# ### 4.3 Decision Tree (tune `max_depth`, `min_samples_split`, `criterion`)

# %%
dt = DecisionTreeClassifier(random_state=40)

param_grid_dt = {
    "max_depth": list(range(2, 21)),
    "min_samples_split": [2, 5, 10, 20],
    "criterion": ["gini", "entropy", "log_loss"]
}

gs_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring="f1_macro", n_jobs=-1)
gs_dt.fit(X_train_scaled, y_train)

print("Best DT params:", gs_dt.best_params_)
best_dt = gs_dt.best_estimator_

acc_dt, f1_dt = evaluate_model("Decision Tree", best_dt, X_test_scaled, y_test)


# %% [markdown]
# ### 4.4 Random Forest (tune `n_estimators`, `max_depth`, `min_samples_split`)

# %%
rf = RandomForestClassifier(random_state=40)

param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 3, 5, 8, 12],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

gs_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1_macro", n_jobs=-1)
gs_rf.fit(X_train_scaled, y_train)

print("Best RF params:", gs_rf.best_params_)
best_rf = gs_rf.best_estimator_

acc_rf, f1_rf = evaluate_model("Random Forest", best_rf, X_test_scaled, y_test)


# %% [markdown]
# ## 5) Conclusion: Compare best models

# %%
results = pd.DataFrame({
    "model": ["KNN", "SVM", "Decision Tree", "Random Forest"],
    "accuracy": [acc_knn, acc_svm, acc_dt, acc_rf],
    "macro_f1": [f1_knn, f1_svm, f1_dt, f1_rf],
}).sort_values(by="macro_f1", ascending=False)

results


# %%
plt.figure(figsize=(10, 3), dpi=140)
plt.barh(results["model"], results["macro_f1"])
plt.xlabel("Macro F1 (higher is better)")
plt.title("Model Comparison (Macro F1)")
plt.xlim(0, 1.0)
plt.show()


# %% [markdown]
# ### Save trained model (optional)
# If you want to deploy later, you can save the scaler + best model with `joblib`.

# %%
import joblib

best_row = results.iloc[0]
print("Best model by Macro F1:", best_row["model"])

best_model_map = {
    "KNN": best_knn,
    "SVM": best_svm,
    "Decision Tree": best_dt,
    "Random Forest": best_rf
}

final_model = best_model_map[best_row["model"]]

joblib.dump({"scaler": scaler, "label_encoder": le, "model": final_model}, "malnutrition_model.joblib")
print("Saved -> malnutrition_model.joblib")



