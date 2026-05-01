# -*- coding: utf-8 -*-
"""
MalnutriScanAI - Ultimate Master File
Halaman utama kini lebih modular: navigation, dashboard, prediction, analytics, dataset di file terpisah.
"""

import streamlit as st
import streamlit.components.v1 as components
import warnings
from navigation import render_top_nav
from dashboard import show_dashboard
from prediction import show_prediction
from analytics import show_analytics
from dataset import show_dataset

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MalnutriScan AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #F8FAFC; }
    /* --- MEMBUNUH SEMUA PENGGANJAL ATAS STREAMLIT --- */
    [data-testid="stSidebar"] { display: none !important; }
    
    header[data-testid="stHeader"] { 
        display: none !important; 
        height: 0px !important; 
        min-height: 0px !important; 
    }
    
    .main .block-container, 
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] { 
        padding-top: 1rem !important; /* Beri sedikit ruang napas di atas */
        padding-bottom: 2rem !important; 
        max-width: 1200px !important; 
        margin-top: -45px !important; /* Tarikan dikurangi agar tidak kepotong */
    }

    /* Membunuh jarak kosong otomatis dari div pembungkus Streamlit */
    div.element-container:nth-child(1) div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 20px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 24px !important;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 20px;
    }

    h1, h2, h3 { color: #0F172A !important; font-weight: 800 !important; }
    h4 { color: #1E293B !important; font-weight: 700 !important; font-size: 18px !important; margin-bottom: 5px !important; }
    .subtitle { color: #64748B; font-size: 14px; margin-bottom: 20px; }
    .section-title { font-size: 20px; font-weight: 800; color: #0F172A; margin: 30px 0 15px 0; }

    svg { vertical-align: middle; display: inline-block; flex-shrink: 0; }
    div.stButton > button p {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        margin: 0 !important;
        white-space: nowrap !important;
        flex-wrap: nowrap !important;
    }

    div.stButton > button { border-radius: 10px !important; font-weight: 600 !important; font-size: 14px !important; padding: 10px 16px !important; border: none !important; }
    div.stButton > button[kind="primary"] { background-color: #3B82F6 !important; color: white !important; }
    div.stButton > button[kind="secondary"] { background-color: transparent !important; color: #64748B !important; border: 1px solid #E2E8F0 !important; }
    div.stButton > button[kind="secondary"]:hover { background-color: #F8FAFC !important; color: #0F172A !important; }

    .hero-banner { background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); border-radius: 24px; padding: 50px 40px; color: white; margin-bottom: 30px; box-shadow: 0 15px 30px -5px rgba(139, 92, 246, 0.3); }
    .hero-badge { border: 1px solid rgba(255,255,255,0.3); background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; gap:6px; margin-bottom: 20px; }
    .hero-title { font-size: 38px; font-weight: 800; line-height: 1.2; margin-bottom: 16px; color: white !important; }
    .hero-subtitle { font-size: 16px; color: rgba(255,255,255,0.9); max-width: 700px; margin-bottom: 30px; line-height: 1.5; }
    .hero-btn { background: white; color: #3B82F6; padding: 10px 24px; border-radius: 12px; font-weight: 700; font-size: 14px; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer; }

    .perf-card { background: white; border-radius: 16px; padding: 20px; border: 1px solid #E2E8F0; height: 100%; }
    .perf-title { font-size: 14px; font-weight: 700; margin-bottom: 15px; }
    .perf-value { font-size: 36px; font-weight: 800; margin-bottom: 5px; }
    .perf-desc { font-size: 12px; display: flex; align-items: center; gap: 5px; font-weight: 600; }
    .c-green { border-color: #86EFAC; } .t-green { color: #10B981; }
    .c-blue { border-color: #93C5FD; } .t-blue { color: #3B82F6; }
    .c-purple { border-color: #D8B4FE; } .t-purple { color: #A855F7; }
    .c-orange { border-color: #FCD34D; } .t-orange { color: #F59E0B; }

    .stat-box { border-radius: 12px; padding: 16px; border: 1px solid #E2E8F0; text-align: left; }
    .stat-box.b-blue { border-color: #BFDBFE; background: #F8FAFC; } .stat-box.b-green { border-color: #BBF7D0; background: #F0FDF4; }
    .stat-box.b-yellow { border-color: #FEF08A; background: #FEFCE8; } .stat-box.b-orange { border-color: #FED7AA; background: #FFF7ED; }
    .stat-label { font-size: 12px; font-weight: 700; color: #64748B; margin-bottom: 5px; }
    .stat-val { font-size: 26px; font-weight: 800; }
    .summary-text { font-size: 14px; font-weight: 600; color: #64748B; margin-top: 15px; }
    .summary-val { font-size: 20px; font-weight: 800; color: #0F172A; }
    .t-red { color: #EF4444; }

    .feature-card { background: white; border-radius: 16px; padding: 24px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); height: 100%; }
    .icon-box { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px; }
    .ib-blue { background: #EFF6FF; color: #3B82F6; } .ib-green { background: #F0FDF4; color: #10B981; } .ib-purple { background: #FAF5FF; color: #A855F7; }

    .akpi-card { background: white; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .akpi-card.b-blue { border-top: 4px solid #3B82F6; } .akpi-card.b-green { border-top: 4px solid #10B981; }
    .akpi-card.b-orange { border-top: 4px solid #F59E0B; } .akpi-card.b-purple { border-top: 4px solid #A855F7; }
    .akpi-header { display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 700; margin-bottom: 15px; }
    .akpi-val { font-size: 32px; font-weight: 800; color: #0F172A; margin-bottom: 2px; }
    .akpi-sub { font-size: 12px; font-weight: 500; }

    .metric-container { margin-bottom: 20px; }
    .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .metric-title { font-size: 13px; font-weight: 600; color: #1E293B; }
    .metric-badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 700; color: white; }
    .bg-green { background-color: #10B981; } .bg-blue { background-color: #3B82F6; } .bg-purple { background-color: #A855F7; } .bg-orange { background-color: #F59E0B; }
    .progress-track { width: 100%; height: 6px; background-color: #F1F5F9; border-radius: 10px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 10px; }
    .metric-desc { font-size: 11px; color: #94A3B8; margin-top: 6px; }

    .feature-pill { border: 1px solid #E2E8F0; background: white; border-radius: 15px; padding: 6px 14px; font-size: 12px; font-weight: 600; color: #334155; display: inline-block; margin: 5px 5px 0 0; }
    .feature-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; margin-top: 15px; }

    .insights-box { background-color: #F0F4FF; border: 1px solid #D0E0FF; border-radius: 16px; padding: 24px; margin-top: 20px; }
    .insights-box ul { margin: 0; padding-left: 20px; color: #475569; font-size: 14px; line-height: 1.8; }
    .insights-box li { margin-bottom: 8px; }

    .btn-export { background: #0F172A; color: white; padding: 10px 20px; border-radius: 10px; font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; float: right; cursor: pointer; }
    .data-card { background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; display: flex; flex-direction: column; justify-content: center; }
    .dc-title { font-size: 13px; color: #64748B; font-weight: 600; display: flex; align-items: center; gap: 6px; margin-bottom: 15px; }
    .dc-val { font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 2px; }
    .dc-sub { font-size: 12px; color: #94A3B8; }
    .kaggle-pill { background: #3B82F6; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; display: inline-block; margin-bottom: 8px; }

    .filter-btn { border: 1px solid #E2E8F0; color: #64748B; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; background: white; cursor: pointer; display: inline-flex; align-items: center; gap: 4px; }
    .filter-btn.active { background: #0F172A; color: white; border-color: #0F172A; }

    .custom-table-container { overflow-x: auto; margin-top: 15px; }
    .custom-table { width: 100%; border-collapse: collapse; min-width: 900px; }
    .custom-table th { border-bottom: 2px solid #E2E8F0; padding: 12px 10px; text-align: left; font-size: 12px; font-weight: 700; color: #0F172A; white-space: nowrap; }
    .custom-table td { border-bottom: 1px solid #F1F5F9; padding: 14px 10px; font-size: 13px; color: #334155; }
    .s-pill { border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 700; display: inline-block; }
    .p-normal { border: 1px solid #10B981; color: #10B981; } .p-mild { border: 1px solid #EAB308; color: #EAB308; }
    .p-moderate { border: 1px solid #F97316; color: #F97316; } .p-severe { border: 1px solid #EF4444; color: #EF4444; background: #FEF2F2; }

    .dict-item { border: 1px solid #F1F5F9; background: #FAFAF9; border-radius: 12px; padding: 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .dict-title { font-weight: 700; font-size: 14px; color: #0F172A; }
    .dict-desc { font-size: 12px; color: #64748B; margin-top: 4px; }
    .badge-type { background: #FFFFFF; border: 1px solid #E2E8F0; color: #475569; padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; }

    .input-label { font-weight: 600; color: #0F172A; font-size: 14px; margin-bottom: 8px; display: block; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background-color: #F8FAFC !important; border-radius: 8px !important; border: 1px solid #E2E8F0 !important; }
    .symptom-container { border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 24px; margin-top: 10px; background-color: #F8FAFC; }

    .predict-btn-wrapper button { background-color: #010B13 !important; color: white !important; border-radius: 10px !important; height: 50px !important; font-size: 15px !important; box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important; margin-top: 10px; }
    .predict-btn-wrapper button:hover { background-color: #1E293B !important; }

    .res-hero { border-radius: 24px; padding: 40px 20px; text-align: center; margin-bottom: 20px; border: 2px solid; }
    .res-hero.moderate { background-color: #FFFBEB; border-color: #FEF08A; }
    .res-hero.normal { background-color: #F0FDF4; border-color: #BBF7D0; }
    .res-hero.severe { background-color: #FEF2F2; border-color: #FECACA; }
    .res-icon { width: 64px; height: 64px; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 15px; }
    .res-icon.moderate { background-color: #F59E0B; } .res-icon.normal { background-color: #10B981; } .res-icon.severe { background-color: #EF4444; }
    .res-title { font-size: 32px; font-weight: 800; color: #0F172A; margin-bottom: 5px; }
    .res-sub { color: #64748B; font-size: 15px; margin-bottom: 25px; }
    .res-pills-container { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
    .res-pill { background: white; border: 1px solid #E2E8F0; padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 700; color: #0F172A; }
    .res-pill.filled.moderate { background: #F59E0B; color: white; border-color: #F59E0B; }
    .res-pill.filled.normal { background: #10B981; color: white; border-color: #10B981; }
    .res-pill.filled.severe { background: #EF4444; color: white; border-color: #EF4444; }
    .res-main-bar { width: 300px; height: 8px; background-color: #E2E8F0; border-radius: 10px; margin: 0 auto; overflow: hidden; }
    .res-main-bar-fill { height: 100%; background-color: #0F172A; border-radius: 10px; }
    .risk-row { display: flex; flex-direction: column; margin-bottom: 15px; }
    .risk-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; }
    .risk-name { font-weight: 700; color: #0F172A; } .risk-val { color: #64748B; font-weight: 500; }
    .risk-pts { font-weight: 800; color: #0F172A; }
    .risk-bar-bg { width: 100%; height: 6px; background-color: #F1F5F9; border-radius: 4px; overflow: hidden; position: relative; }
    .risk-bar-fill { height: 100%; background-color: #94A3B8; border-radius: 4px; }
    .risk-pct { font-size: 10px; color: #64748B; text-align: right; margin-top: 2px; font-weight: 600; }
    .rec-item { border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; display: flex; align-items: center; gap: 12px; font-size: 13px; color: #334155; background: white; }
    .disclaimer { display: flex; gap: 10px; font-size: 11px; color: #94A3B8; margin-top: 20px; line-height: 1.5; padding: 0 10px; }

    .footer { text-align: center; color: #94A3B8; font-size: 12px; margin-top: 50px; font-weight: 500; border-top: 1px solid #E2E8F0; padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# INISIALISASI SESSION STATE
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'dataset_filter' not in st.session_state:
    st.session_state.dataset_filter = 'All'
if 'pred_done' not in st.session_state:
    st.session_state.pred_done = False
if 'pred_data' not in st.session_state:
    st.session_state.pred_data = {}

# Inisialisasi last_page diletakkan di sini, tetapi eksekusi JS auto-scroll-nya dipindah ke bawah
if 'last_page' not in st.session_state:
    st.session_state.last_page = st.session_state.page


def main():
    # Render navigasi atas WAJIB menjadi elemen pertama agar jarak tidak membesar
    render_top_nav()

    # Logika routing halaman
    if st.session_state.page == 'Dashboard':
        show_dashboard()
    elif st.session_state.page == 'Prediksi':
        show_prediction()
    elif st.session_state.page == 'Analytics':
        show_analytics()
    elif st.session_state.page == 'Dataset':
        show_dataset()

    st.markdown(
        """<div class="footer">© 2026 MalnutriScan AI - Advanced Malnutrition Prediction System<br>Powered by Machine Learning - Dataset: Kaggle Nutrition Dataset - Model Accuracy 94%</div>""",
        unsafe_allow_html=True,
    )

    # --- SISTEM CERDAS PENARIK LAYAR KE ATAS (VERSI BRUTAL) DIPINDAH KE SINI ---
    # Dengan diletakkan di akhir fungsi, kode ini tidak akan mengganggu layout nth-child(1) dari Navbar!
    if st.session_state.page != st.session_state.last_page:
        js_scroll_up = '''
        <script>
            function forceScrollTop() {
                // 1. Tarik layar utama window
                window.parent.scrollTo(0, 0);
                
                // 2. Cari SEMUA kontainer pembungkus bawaan Streamlit (dari versi lama sampai terbaru)
                var parentDoc = window.parent.document;
                var containers = parentDoc.querySelectorAll('section.main, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], [data-testid="stMain"], .stApp');
                
                containers.forEach(function(c) {
                    c.scrollTop = 0;
                });
            }
            
            // Eksekusi beruntun untuk membantai loading lambat Streamlit!
            forceScrollTop();               // Tembakan pertama (Instan)
            setTimeout(forceScrollTop, 50); // Tembakan kedua (50ms)
            setTimeout(forceScrollTop, 150);// Tembakan ketiga (150ms)
            setTimeout(forceScrollTop, 300);// Tembakan terakhir (300ms - pasti kena!)
        </script>
        '''
        components.html(js_scroll_up, height=0)
        
        # Catat halaman yang baru dibuka
        st.session_state.last_page = st.session_state.page
    # -----------------------------------------------------------------


if __name__ == "__main__":
    main()