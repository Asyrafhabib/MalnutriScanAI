import os
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from utils import load_svg, get_svg_base64 
from model import load_saved_models

def show_analytics():
    # ==========================================
    # 1. LOAD MODEL & DATA DINAMIS (ANTI-CACHE)
    # ==========================================
    models_dict, le, scaler, numeric_cols = load_saved_models()

    if not models_dict:
        st.warning("Menunggu file model... Pastikan folder 'model' berisi file .joblib")
        return

    dataset_filename = os.path.join('data', 'prediction_history.csv')
    try:
        with open(dataset_filename, 'r', encoding='utf-8') as file:
            df = pd.read_csv(file)
        df = df.drop_duplicates()
    except Exception as e:
        st.error(f"Gagal memuat file history: {e}")
        return

    # Kalkulasi Metrik Kartu KPI
    total_patients = len(df)
    normal_count = len(df[df['nutrition_status'].str.lower() == 'normal'])
    moderate_count = len(df[df['nutrition_status'].str.lower() == 'moderate'])
    severe_count = len(df[df['nutrition_status'].str.lower() == 'severe'])
    normal_pct = (normal_count / total_patients) * 100 if total_patients > 0 else 0
    at_risk_count = moderate_count + severe_count
    avg_bmi = df['bmi'].mean() if 'bmi' in df.columns else 0.0
    avg_muac = df['muac_cm'].mean() if 'muac_cm' in df.columns else 0.0
    avg_age = df['age_months'].mean() if 'age_months' in df.columns else 0.0

    X = df[numeric_cols]
    y_true = le.transform(df["nutrition_status"])
    
    xgb_pred = models_dict['XGBoost'].predict(X)
    acc_val = round(accuracy_score(y_true, xgb_pred) * 100, 1)

    icon_users = load_svg("users-round.svg")
    icon_health = load_svg("trending-up.svg")
    icon_warning = load_svg("triangle-alert.svg")
    icon_activity = load_svg("activity.svg")

    st.markdown("<h3>Analytics Dashboard</h3><p class='subtitle'>Comprehensive insights from prediction history processed with the model pipeline</p>", unsafe_allow_html=True)
    
    # ==========================================
    # 2. RENDER KPI CARDS
    # ==========================================
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="akpi-card b-blue"><div class="akpi-header t-blue"><span>Total Predictions</span><div style="width:16px; height:16px; display:flex;">{icon_users}</div></div><div class="akpi-val t-blue">{total_patients:,}</div><div class="akpi-sub t-blue">history records</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="akpi-card b-green"><div class="akpi-header t-green"><span>Normal</span><div style="width:16px; height:16px; display:flex;">{icon_health}</div></div><div class="akpi-val t-green">{normal_count:,}</div><div class="akpi-sub t-green">{normal_pct:.1f}% of total</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="akpi-card b-orange"><div class="akpi-header t-orange"><span>At Risk</span><div style="width:16px; height:16px; display:flex;">{icon_warning}</div></div><div class="akpi-val t-orange">{at_risk_count:,}</div><div class="akpi-sub t-orange">moderate + severe</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="akpi-card b-purple"><div class="akpi-header t-purple"><span>Avg BMI</span><div style="width:16px; height:16px; display:flex;">{icon_activity}</div></div><div class="akpi-val t-purple">{avg_bmi:.1f}</div><div class="akpi-sub t-purple">Prediction history average</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # 3. GRAFIK DATA DISTRIBUSI
    # ==========================================
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### Malnutrition Status Distribution\n<p class='subtitle'>Breakdown of patient categories</p>", unsafe_allow_html=True)
            status_counts = df['nutrition_status'].value_counts()
            fig1 = px.pie(values=status_counts.values, names=[s.capitalize() for s in status_counts.index], hole=0.5, color_discrete_sequence=['#10B981', '#F97316', '#EF4444'])
            fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig1, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            st.markdown("#### Age Group Analysis\n<p class='subtitle'>Distribution across age months</p>", unsafe_allow_html=True)
            age_bins = [0, 18, 35, 60, 1000]
            age_labels = ['0-18', '19-35', '36-60', '60+']
            df['AgeGroup'] = pd.cut(df['age_months'], bins=age_bins, labels=age_labels, right=True)
            age_counts = df['AgeGroup'].value_counts().reindex(age_labels).fillna(0)
            fig2 = px.bar(x=age_labels, y=age_counts.values, color_discrete_sequence=['#3B82F6'])
            fig2.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=250, xaxis_title="Age (Months)", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown("#### Gender Distribution\n<p class='subtitle'>Patient count by gender</p>", unsafe_allow_html=True)
            gender_counts = df['gender'].value_counts()
            fig3 = px.pie(values=gender_counts.values, names=gender_counts.index, hole=0.5, color_discrete_sequence=['#EC4899', '#06B6D4'])
            fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig3, use_container_width=True)
    
    with c4:
        with st.container(border=True):
            st.markdown("#### MUAC Distribution\n<p class='subtitle'>Mid-Upper Arm Circumference analysis</p>", unsafe_allow_html=True)
            fig4 = px.histogram(df, x='muac_cm', nbins=8, color_discrete_sequence=['#8B5CF6'], labels={'muac_cm': 'MUAC (cm)', 'count': 'Frequency'})
            fig4.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=250, xaxis_title="MUAC (cm)", yaxis_title="Frequency", showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ==========================================
    # 4. DATA EXPLORER (CUSTOM UI PIXEL-PERFECT)
    # ==========================================
    b64_filter = get_svg_base64("funnel.svg") 
    b64_download = get_svg_base64("download.svg")
    b64_search = get_svg_base64("search.svg") 
    
    st.markdown(f"""
    <style>
        .explorer-title {{ color: #64748B; font-size: 15px; font-weight: 600; padding-top: 5px; margin-bottom: 0px; }}
        .explorer-info {{ color: #94A3B8; font-size: 13px; margin-top: 15px; margin-bottom: 5px; }}
        
        div[data-testid="column"] > div {{ padding: 0 !important; }}

        /* --- KOREKSI ALIGNMENT VERTIKAL --- */
        div[data-testid="stTextInput"] {{ transform: translateY(2px) !important; }}
        div[data-testid="stButton"] {{ transform: translateY(-2px) !important; }}

        /* --- DESAIN SEARCH BAR + IKON SVG --- */
        div[data-testid="stTextInput"] div[data-baseweb="input"] {{
            border-radius: 8px !important; 
            height: 34px !important; /* Diperkecil dari 40px */
            min-height: 34px !important; 
            max-height: 34px !important; 
            background-color: #F8FAFC !important; 
            border: 1px solid #E2E8F0 !important; 
            box-sizing: border-box !important; 
        }}
        div[data-testid="stTextInput"] input {{ 
            font-size: 13px !important; /* Huruf sedikit dikecilkan */
            color: #0F172A !important; 
            background-color: transparent !important; 
            line-height: 34px !important; /* Menyesuaikan tinggi baru */
            background-image: url('{b64_search}') !important; 
            background-repeat: no-repeat !important; 
            background-position: 12px center !important; 
            background-size: 14px 14px !important; /* Ikon sedikit dikecilkan */
            padding-left: 34px !important;
        }}

        /* --- DESAIN TOMBOL HITAM (AKTIF) --- */
        div[data-testid="stElementContainer"]:has(.btn-marker-black) + div[data-testid="stElementContainer"] button,
        .element-container:has(.btn-marker-black) + .element-container button {{
            background-color: #0F172A !important; color: #FFFFFF !important; border: 1px solid #0F172A !important; border-radius: 8px !important; font-weight: 600 !important; 
            height: 34px !important; min-height: 34px !important; max-height: 34px !important; 
            width: 100% !important; box-shadow: none !important; padding: 0 10px !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; margin: 0 !important; box-sizing: border-box !important;
        }}
        div[data-testid="stElementContainer"]:has(.btn-marker-black) + div[data-testid="stElementContainer"] button p,
        .element-container:has(.btn-marker-black) + .element-container button p {{ 
            color: #FFFFFF !important; margin: 0 !important; display: flex; align-items: center; font-size: 13px !important; 
        }}
        
        /* --- IKON FILTER KUNCI PERMANEN HANYA DI TOMBOL ALL --- */
        div[data-testid="stElementContainer"]:has(.btn-marker-all) + div[data-testid="stElementContainer"] button p::before,
        .element-container:has(.btn-marker-all) + .element-container button p::before {{
            content: ''; display: inline-block !important; width: 13px !important; height: 13px !important; margin-right: 6px !important; background-color: currentColor !important; -webkit-mask: url('{b64_filter}') no-repeat center / contain !important; mask: url('{b64_filter}') no-repeat center / contain !important;
        }}

        /* --- DESAIN TOMBOL PUTIH (TIDAK AKTIF) --- */
        div[data-testid="stElementContainer"]:has(.btn-marker-white) + div[data-testid="stElementContainer"] button,
        .element-container:has(.btn-marker-white) + .element-container button {{
            background-color: #FFFFFF !important; color: #64748B !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; font-weight: 600 !important; 
            height: 34px !important; min-height: 34px !important; max-height: 34px !important; 
            width: 100% !important; box-shadow: none !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; margin: 0 !important; box-sizing: border-box !important;
        }}
        div[data-testid="stElementContainer"]:has(.btn-marker-white) + div[data-testid="stElementContainer"] button p,
        .element-container:has(.btn-marker-white) + .element-container button p {{ 
            color: #64748B !important; margin: 0 !important; font-size: 13px !important; 
        }}
        
        /* Ikon Download untuk Export */
        div[data-testid="stElementContainer"]:has(.btn-marker-export) + div[data-testid="stElementContainer"] button p::before,
        .element-container:has(.btn-marker-export) + .element-container button p::before {{
            content: ''; display: inline-block !important; width: 14px !important; height: 14px !important; margin-right: 8px !important; background-color: currentColor !important; -webkit-mask: url('{b64_download}') no-repeat center / contain !important; mask: url('{b64_download}') no-repeat center / contain !important;
        }}
        
        /* Ikon Filter KHUSUS untuk Tombol All (Selalu Muncul) */
        div[data-testid="stElementContainer"]:has(.btn-marker-all) + div[data-testid="stElementContainer"] button p::before,
        .element-container:has(.btn-marker-all) + .element-container button p::before {{
            content: ''; display: inline-block !important; width: 13px !important; height: 13px !important; margin-right: 6px !important; background-color: currentColor !important; -webkit-mask: url('{b64_filter}') no-repeat center / contain !important; mask: url('{b64_filter}') no-repeat center / contain !important;
        }}

        /* --- DESAIN TOMBOL PUTIH (NORMAL, MODERATE, SEVERE) --- */
        div[data-testid="stElementContainer"]:has(.btn-marker-white) + div[data-testid="stElementContainer"] button,
        .element-container:has(.btn-marker-white) + .element-container button {{
            background-color: #FFFFFF !important; color: #64748B !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; font-weight: 600 !important; height: 40px !important; min-height: 40px !important; max-height: 40px !important; width: 100% !important; box-shadow: none !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; margin: 0 !important;
        }}
        div[data-testid="stElementContainer"]:has(.btn-marker-white) + div[data-testid="stElementContainer"] button p,
        .element-container:has(.btn-marker-white) + .element-container button p {{ color: #64748B !important; margin: 0 !important; }}
        
        /* --- FITUR SCROLLING TABEL & STICKY HEADER --- */
        .custom-table-container {{
            max-height: 480px !important; /* Setara dengan sekitar 10 baris */
            overflow-y: auto !important; /* Memunculkan scroll bar vertikal jika lebih dari batas */
            border: 1px solid #E2E8F0; /* Bingkai luar agar rapi saat di-scroll */
            border-radius: 8px;
        }}
        
        .custom-table th {{
            text-transform: capitalize !important;
            white-space: nowrap !important; 
            position: sticky !important; /* Header menempel di atas */
            top: 0 !important; 
            background-color: #F8FAFC !important; /* Warna solid agar teks di bawahnya tertutup saat scroll */
            z-index: 10 !important;
            box-shadow: inset 0 -1px 0 #E2E8F0 !important; /* Garis bawah header */
        }}

        /* Memaksa semua elemen dalam satu baris untuk sejajar di tengah */
        div[data-testid="stHorizontalBlock"] {{
            align-items: center !important;
        }}

        /* --- MENGHANCURKAN JARAK GAIB MARKER STREAMLIT --- */
        /* Mencegah st.markdown (marker) memakan ruang vertikal dan mendorong tombol ke bawah */
        div[data-testid="stElementContainer"]:has(.btn-marker-black),
        div[data-testid="stElementContainer"]:has(.btn-marker-white),
        div[data-testid="stElementContainer"]:has(.btn-marker-export),
        .element-container:has(.btn-marker-black),
        .element-container:has(.btn-marker-white),
        .element-container:has(.btn-marker-export) {{
            height: 0px !important;
            min-height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
            display: none !important;
        }}

        /* --- MEMBUNUH LABEL BAWAAN TEXT INPUT YANG MAKAN TEMPAT --- */
        /* Streamlit selalu menyiapkan ruang untuk teks label meskipun kita sembunyikan */
        div[data-testid="stTextInput"] label {{
            display: none !important;
            height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
        }}

        /* Menghapus margin bawaan input dan tombol */
        div[data-testid="stTextInput"], div[data-testid="stButton"] {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        /* --- MENGECILKAN KHUSUS TOMBOL EXPORT --- */
        div[data-testid="stElementContainer"]:has(.btn-marker-export) + div[data-testid="stElementContainer"] button,
        .element-container:has(.btn-marker-export) + .element-container button {{
            height: 34px !important; 
            min-height: 34px !important; 
            max-height: 34px !important; 
            width: fit-content !important; /* Lebar hanya menyesuaikan teks */
            padding: 0 16px !important;
            float: right !important; /* Dorong tombol ke pojok kanan */
        }}
        
        /* Mengecilkan ukuran teks pada tombol export */
        div[data-testid="stElementContainer"]:has(.btn-marker-export) + div[data-testid="stElementContainer"] button p,
        .element-container:has(.btn-marker-export) + .element-container button p {{
            font-size: 13px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        # --- BARIS 1: JUDUL & TOMBOL EXPORT ---
        col_title, col_export = st.columns([4, 1.2])
        with col_title:
            st.markdown("<div class='explorer-title'><h4 style='color: #0F172A; margin-bottom: 15px;'>Prediction History</h4></div>", unsafe_allow_html=True)
        with col_export:
            df_export = df.copy()
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.markdown("<div class='btn-marker-black btn-marker-export' style='display:none;'></div>", unsafe_allow_html=True)
            st.download_button("Export CSV", data=csv_data, file_name="Prediction History.csv", mime="text/csv", use_container_width=True)
            
        st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            
        # State Management untuk Tombol Filter
        if 'active_filter' not in st.session_state:
            st.session_state.active_filter = "All"
            
        # --- BARIS 2: SEARCH & FILTER ---
        c_search, c_all, c_norm, c_mod, c_sev = st.columns([3, 0.7, 0.8, 0.9, 0.8], gap="small")
        
        with c_search:
            search_query = st.text_input("Search", placeholder="Search by ID or Age...", label_visibility="collapsed")
            
        # --- LOGIKA WARNA TOMBOL DINAMIS ---
        # Ambil status filter yang sedang aktif saat ini
        active_f = st.session_state.active_filter
        
        with c_all:
            # Jika aktif, pakai marker hitam. Jika tidak, pakai putih.
            marker_all = "btn-marker-black" if active_f == "All" else "btn-marker-white"
            st.markdown(f"<div class='{marker_all} btn-marker-all'></div>", unsafe_allow_html=True)
            if st.button("All", key="btn_f_all", use_container_width=True):
                st.session_state.active_filter = "All"
                st.rerun()
                
        with c_norm:
            marker_norm = "btn-marker-black" if active_f == "Normal" else "btn-marker-white"
            st.markdown(f"<div class='{marker_norm}'></div>", unsafe_allow_html=True)
            if st.button("Normal", key="btn_f_norm", use_container_width=True):
                st.session_state.active_filter = "Normal"
                st.rerun()
                
        with c_mod:
            marker_mod = "btn-marker-black" if active_f == "Moderate" else "btn-marker-white"
            st.markdown(f"<div class='{marker_mod}'></div>", unsafe_allow_html=True)
            if st.button("Moderate", key="btn_f_mod", use_container_width=True):
                st.session_state.active_filter = "Moderate"
                st.rerun()
                
        with c_sev:
            marker_sev = "btn-marker-black" if active_f == "Severe" else "btn-marker-white"
            st.markdown(f"<div class='{marker_sev}'></div>", unsafe_allow_html=True)
            if st.button("Severe", key="btn_f_sev", use_container_width=True):
                st.session_state.active_filter = "Severe"
                st.rerun()
                
        # --- LOGIKA FILTERING DATA ---
        df_display = df.copy()
        
        if 'AgeGroup' in df_display.columns:
            df_display = df_display.drop(columns=['AgeGroup'])
        
        if search_query:
            mask = df_display.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
            df_display = df_display[mask]
            
        if st.session_state.active_filter != "All":
            mask = df_display['nutrition_status'].astype(str).str.contains(st.session_state.active_filter, case=False, na=False)
            df_display = df_display[mask]
            
        total_filtered = len(df_display)
        # BUKA BATAS MAKSIMAL MENJADI 100 BARIS AGAR SCROLL TERLIHAT
        display_limit = min(100, total_filtered) 
        
        if total_filtered == total_patients:
            st.markdown(f"<div class='explorer-info'>Showing {display_limit} of {total_patients:,} records</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='explorer-info'>Showing {display_limit} of {total_filtered:,} filtered records</div>", unsafe_allow_html=True)
        
        # --- RENDER TABEL HTML DENGAN HEADER RAPI ---
        # 1. Buat kamus nama header kustom di sini
        nama_kustom = {
            "timestamp": "Prediction Time",
            "age_months": "Age (Months)",
            "gender": "Gender",
            "weight_kg": "Weight (kg)",
            "height_cm": "Height (cm)",
            "muac_cm": "MUAC (cm)",
            "bmi": "BMI Score",
            "nutrition_status": "Nutrition Status"
        }
        
        # 2. Render header menggunakan kamus (kalau kolomnya nggak ada di kamus, pakai nama aslinya)
        headers_html = "".join([f"<th>{nama_kustom.get(col, col.replace('_', ' ').title())}</th>" for col in df_display.columns])
        rows_html = ""
        
        if total_filtered == 0:
            rows_html = f"<tr><td colspan='{len(df_display.columns)}' style='text-align:center; padding:30px; color:#94A3B8;'>No records found.</td></tr>"
        else:
            for _, row in df_display.head(display_limit).iterrows():
                row_html = "<tr>"
                for col in df_display.columns:
                    val = row[col]
                    if str(col).lower() == 'nutrition_status':
                        val_str = str(val).capitalize()
                        pill_class = "s-pill p-normal"
                        if "moderate" in val_str.lower(): pill_class = "s-pill p-moderate"
                        elif "severe" in val_str.lower(): pill_class = "s-pill p-severe"
                        row_html += f'<td><span class="{pill_class}">{val_str}</span></td>'
                    else:
                        row_html += f'<td>{val}</td>'
                row_html += "</tr>"
                rows_html += row_html
        
        st.markdown(f"""
        <div class="custom-table-container">
            <table class="custom-table">
                <thead><tr>{headers_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # 5. KEY INSIGHTS
    # ==========================================
    severe_pct = round((len(df[df['nutrition_status'].str.lower() == 'severe']) / total_patients) * 100, 1) if total_patients > 0 else 0
    
    st.markdown(
        f"""<div class="insights-box"><h4 style="color: #0F172A; margin-bottom: 15px;">XGBoost Algorithm Insights</h4><ul><li><b>{severe_pct}%</b> of patients in this dataset are classified as Severe Risk.</li><li>The <b>XGBoost Algorithm</b> maintains an accuracy of <b>{acc_val}%</b> on this specific batch of records.</li><li>Average BMI across the population stands at <b>{avg_bmi:.1f}</b>.</li><li>MUAC and BMI are identified as the most significant predictors for malnutrition status.</li></ul></div>""",
        unsafe_allow_html=True,
    )