import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_svg
from model import load_and_train_model  # <-- IMPORT FILE MODEL

def show_analytics():
    # ==========================================
    # 1. LOAD MODEL & DATA DINAMIS
    # ==========================================
    models, scaler, le, df, numeric_cols = load_and_train_model()

    # Kalkulasi Metrik Kartu KPI
    total_patients = len(df)
    healthy_count = len(df[df['nutrition_status'].str.lower() == 'normal'])
    healthy_pct = (healthy_count / total_patients) * 100 if total_patients > 0 else 0
    at_risk_count = total_patients - healthy_count
    avg_bmi = df['bmi'].mean() if 'bmi' in df.columns else 0.0

    # Kalkulasi Metrik Machine Learning Setup
    X = df[numeric_cols]
    y = df["nutrition_status_encoded"]
    
    # Re-split dengan seed yang persis sama di model.py untuk evaluasi test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaler khusus untuk Logistic Regression
    X_test_scaled = scaler.transform(X_test)
    
    # Simpan akurasi XGBoost untuk Kesimpulan Akhir (Insights)
    xgb_pred = models['XGBoost'].predict(X_test)
    acc_val = round(accuracy_score(y_test, xgb_pred) * 100, 1)

    icon_users = load_svg("users-round.svg")
    icon_health = load_svg("trending-up.svg")
    icon_warning = load_svg("triangle-alert.svg")
    icon_activity = load_svg("activity.svg")

    st.markdown("<h1>Analytics Dashboard</h1><p class='subtitle'>Comprehensive insights from the malnutrition dataset</p>", unsafe_allow_html=True)
    
    # ==========================================
    # 2. RENDER KPI CARDS
    # ==========================================
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="akpi-card b-blue"><div class="akpi-header t-blue"><span>Total Patients</span><div style="width:16px; height:16px; display:flex;">{icon_users}</div></div><div class="akpi-val t-blue">{total_patients:,}</div><div class="akpi-sub t-blue">in training dataset</div></div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="akpi-card b-green"><div class="akpi-header t-green"><span>Healthy Status</span><div style="width:16px; height:16px; display:flex;">{icon_health}</div></div><div class="akpi-val t-green">{healthy_count:,}</div><div class="akpi-sub t-green">{healthy_pct:.1f}% of total</div></div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class="akpi-card b-orange"><div class="akpi-header t-orange"><span>At Risk</span><div style="width:16px; height:16px; display:flex;">{icon_warning}</div></div><div class="akpi-val t-orange">{at_risk_count:,}</div><div class="akpi-sub t-orange">Requires intervention</div></div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class="akpi-card b-purple"><div class="akpi-header t-purple"><span>Avg BMI</span><div style="width:16px; height:16px; display:flex;">{icon_activity}</div></div><div class="akpi-val t-purple">{avg_bmi:.1f}</div><div class="akpi-sub t-purple">Dataset average</div></div>""",
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # 3. GRAFIK DATA ASLI
    # ==========================================
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            st.markdown("#### Malnutrition Status Distribution\n<p class='subtitle'>Breakdown of patient categories in dataset</p>", unsafe_allow_html=True)
            
            status_counts = df['nutrition_status'].value_counts()
            labels = [s.capitalize() for s in status_counts.index]
            values = status_counts.values.tolist()
            
            fig1 = px.pie(
                values=values,
                names=labels,
                hole=0.5,
                color_discrete_sequence=['#10B981', '#F97316', '#EF4444'], # Normal, Moderate, Severe
            )
            fig1.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=250,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig1, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            st.markdown("#### Age Group Analysis\n<p class='subtitle'>Distribution across age groups</p>", unsafe_allow_html=True)
            
            age_bins = [0, 18, 35, 60, 1000]
            age_labels = ['0-18', '19-35', '36-60', '60+']
            df['AgeGroup'] = pd.cut(df['age_months'], bins=age_bins, labels=age_labels, right=True)
            age_counts = df['AgeGroup'].value_counts().reindex(age_labels).fillna(0)
            
            fig3 = px.bar(x=age_labels, y=age_counts.values, color_discrete_sequence=['#F59E0B'])
            fig3.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=250, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # 4. MODEL METRICS (8 MODELS)
    # ==========================================
    with st.container(border=True):
        st.markdown("#### AI Models Performance Metrics\n<p class='subtitle'>Detailed performance analysis of all 8 trained prediction models</p>", unsafe_allow_html=True)
        
        # --- A. PRE-KALKULASI SEMUA MODEL ---
        model_names = list(models.keys())
        calculated_metrics = {}
        chart_data = []

        # Kamus untuk merapikan nama model
        display_names = {
            "LogisticRegression": "Logistic Regression",
            "RandomForest": "Random Forest",
            "XGBoost": "XGBoost", # Menandai model terbaik
            "CatBoost": "CatBoost",
            "SVM": "SVM",
            "KNN": "KNN",
            "NaiveBayes": "Naive Bayes",
            "DecisionTree": "Decision Tree"
        }

        for m_name in model_names:
            model_obj = models[m_name]
            
            # Prediksi sesuai tipe model
            if m_name == "LogisticRegression":
                y_pred = model_obj.predict(X_test_scaled)
            elif m_name == "CatBoost":
                y_pred = model_obj.predict(X_test).flatten().astype(int)
            else:
                y_pred = model_obj.predict(X_test)
                
            # Hitung Skor
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            
            # Simpan ke dictionary untuk Tab dan List untuk Chart
            calculated_metrics[m_name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
            chart_data.append({"Model": display_names.get(m_name, m_name), "Accuracy": acc})

        # --- B. RENDER LEADERBOARD CHART (KOGNITIF UX) ---
        # Urutkan data berdasarkan akurasi
        df_chart = pd.DataFrame(chart_data).sort_values(by="Accuracy", ascending=True)
        
        fig_bar = px.bar(
            df_chart, 
            x="Accuracy", 
            y="Model", 
            orientation='h', 
            text_auto='.1f',
            color="Accuracy", 
            color_continuous_scale="Teal" # Warna estetik yang menyatu dengan UI
        )
        fig_bar.update_layout(
            margin=dict(t=10, b=10, l=0, r=0), 
            height=280, 
            showlegend=False, 
            xaxis_title="Accuracy (%)", 
            yaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("<hr style='margin: 10px 0 20px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

        # --- C. RENDER TAB DETAIL MODEL ---
        def make_progress(title, val, desc, color_class):
            return (
                f"""
                <div class="metric-container"><div class="metric-header"><span class="metric-title">{title}</span><span class="metric-badge {color_class}">{val:.1f}%</span></div><div class="progress-track"><div class="progress-fill {color_class}" style="width: {val}%;"></div></div><div class="metric-desc">{desc}</div></div>
                """
            )
            
        # UI Tabs dengan nama yang sudah dirapikan
        tab_titles = [display_names.get(m, m) for m in model_names]
        tabs = st.tabs(tab_titles)

        # Loop render progress bar di tiap tab (menggunakan data pre-kalkulasi)
        for i, m_name in enumerate(model_names):
            with tabs[i]:
                metrics = calculated_metrics[m_name]
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                
                with p_col1:
                    st.markdown(make_progress("Accuracy", metrics["acc"], "Overall correctness of predictions", "bg-green"), unsafe_allow_html=True)
                with p_col2:
                    st.markdown(make_progress("Precision", metrics["prec"], "Accuracy of positive predictions", "bg-blue"), unsafe_allow_html=True)
                with p_col3:
                    st.markdown(make_progress("Recall", metrics["rec"], "Detection of actual positives", "bg-purple"), unsafe_allow_html=True)
                with p_col4:
                    st.markdown(make_progress("F1 Score", metrics["f1"], "Harmonic mean of precision & recall", "bg-orange"), unsafe_allow_html=True)

        # --- D. MODEL FEATURES (DIPERBAIKI AGAR TIDAK BOCOR DARI CONTAINER) ---
        st.markdown(
            """
            <div class="feature-box" style="margin-top: 30px; margin-bottom: 15px;">
                <div style="font-size: 13px; font-weight: 700; color: #1E293B; margin-bottom: 8px;">Used Model Features (5)</div>
                <div>
                    <span class="feature-pill">Age (Months)</span>
                    <span class="feature-pill">Weight (kg)</span>
                    <span class="feature-pill">Height (cm)</span>
                    <span class="feature-pill">MUAC (cm)</span>
                    <span class="feature-pill">BMI</span>
                </div>
            </div>
            <!-- Pengganjal transparan agar tinggi container Streamlit tereksekusi sempurna -->
            <div style="height: 10px;"></div>
            """,
            unsafe_allow_html=True,
        )
    
    # ==========================================
    # 5. KEY INSIGHTS
    # ==========================================
    severe_pct = round((len(df[df['nutrition_status'].str.lower() == 'severe']) / total_patients) * 100, 1)
    
    st.markdown(
        f"""<div class="insights-box"><h4 style="color: #0F172A; margin-bottom: 15px;">Key Insights</h4><ul><li><b>{severe_pct}%</b> of patients show severe malnutrition requiring immediate medical attention.</li><li>The dataset has a high concentration of patients in the <b>36-60 months</b> age group.</li><li>The XGBoost model achieves the highest performance with <b>{acc_val}% accuracy</b> with balanced precision and recall, making it reliable for screening.</li><li>Average BMI across all patients is <b>{avg_bmi:.1f}</b>.</li></ul></div>""",
        unsafe_allow_html=True,
    )