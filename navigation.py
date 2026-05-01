import streamlit as st

def render_top_nav():
    st.markdown(f"""
        <style>
            /* JURUS PENDORONG: Mencegah konten utama nyangkut di bawah navbar */
            .block-container {{
                padding-top: 90px !important;
            }}

            /* 1. KOTAK PEMBEDA NAVBAR (JURUS PAKU PAYUNG / FIXED) */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) {{
                /* === BARIS AJAIB BARU (PASTI BERHASIL) === */
                position: fixed !important;
                top: 0 !important; /* Memaku tepat di plafon layar */
                left: 0 !important;
                right: 0 !important;
                z-index: 999999 !important; /* Kasta tertinggi, tidak mungkin tertimpa */
                /* ========================================= */


                align-items: center !important; 
                background-color: #FFFFFF !important; 
                border-bottom: 1px solid #E2E8F0 !important; 
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important; 
                margin: 0 !important; /* Reset margin karena sudah fixed */

                /* Lebar Navbar */
                width: 100vw !important; max-width: 100vw !important;
                padding-top: 15px !important; padding-bottom: 15px !important;
                padding-left: max(1rem, calc(50vw - 600px)) !important; padding-right: max(1rem, calc(50vw - 600px)) !important;
            }}

            /* 2. MERAPIKAN KONTEN DI DALAM NAVBAR */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) > div[data-testid="column"] {{
                display: flex !important; flex-direction: column !important; justify-content: center !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) div.element-container,
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) div.stButton,
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) p {{
                margin: 0 !important; padding: 0 !important;
            }}
                
            /* 3. TOMBOL NAVBAR (DIBUAT LEBIH KECIL & SPESIFIK) */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) div[data-testid="column"] div.stButton > button {{
                height: 34px !important; /* Ukuran lebih ramping */
                border-radius: 8px !important;
                font-size: 13px !important;
                padding: 0 12px !important;
                border: 1px solid #E2E8F0 !important;
                background-color: transparent !important;
                color: #64748B !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }}
            
            /* Teks di dalam tombol */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) button p {{
                font-size: 13px !important;
                margin-top: 2px !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) button:hover {{
                border-color: #3B82F6 !important;
                color: #3B82F6 !important;
                background-color: #F8FAFC !important;
            }}

            /* 4. DESAIN LOGO & TEKS KIRI */
            .nav-logo-container {{ display: flex; align-items: center; gap: 12px; margin-top: -8px !important; }}
            .nav-logo-box {{ width: 44px; height: 44px; background-color: #3B82F6; border-radius: 12px; display: flex; align-items: center; justify-content: center; }}
            .nav-logo-box svg {{ color: white; width: 24px; height: 24px; }}
            .nav-brand {{ font-weight: 800; font-size: 20px; color: #3B82F6; line-height: 1.2; }}
            .nav-brand-sub {{ font-size: 13px; color: #64748B; font-weight: 500; }}

            /* 5. MENGATUR TOMBOL BAWAAN STREAMLIT */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) button {{
                height: 44px !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 14px !important; 
                padding: 0 20px !important; border: 1px solid #E2E8F0 !important; background-color: transparent !important; 
                color: #64748B !important; transition: all 0.2s ease; width: 100% !important;
            }}
                
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) button:hover {{
                border-color: #CBD5E1 !important; background-color: #F8FAFC !important; color: #0F172A !important;
            }}
            
            /* Warna Tombol Saat Aktif */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) div.stButton button[kind="primary"] {{
                background-color: #3B82F6 !important; color: white !important; border: none !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) div.stButton button[kind="primary"]:hover {{
                background-color: #2563EB !important;
            }}

            /* 5. MEMBERSIHKAN SISA-SISA KOTAK HITAM SEBELUMNYA */
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) button p::before,
            div[data-testid="stHorizontalBlock"]:has(.nav-identifier) button p::after {{
                display: none !important; content: none !important; background: transparent !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

    with col1:
        st.markdown('<span class="nav-identifier" style="display:none;"></span><div class="nav-logo-container"><div class="nav-logo-box"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg></div><div><div class="nav-brand">MalnutriScan AI</div><div class="nav-brand-sub">Advanced XGBoost Prediction System</div></div></div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("Dashboard", icon=":material/dashboard:", use_container_width=True, type="primary" if st.session_state.page == 'Dashboard' else "secondary"):
            st.session_state.page = 'Dashboard'
            st.rerun()
            
    with col3:
        if st.button("Prediksi", icon=":material/neurology:", use_container_width=True, type="primary" if st.session_state.page == 'Prediksi' else "secondary"):
            st.session_state.page = 'Prediksi'
            st.rerun()
            
    with col4:
        if st.button("Analytics", icon=":material/bar_chart:", use_container_width=True, type="primary" if st.session_state.page == 'Analytics' else "secondary"):
            st.session_state.page = 'Analytics'
            st.rerun()
            
    with col5:
        if st.button("Dataset", icon=":material/database:", use_container_width=True, type="primary" if st.session_state.page == 'Dataset' else "secondary"):
            st.session_state.page = 'Dataset'
            st.rerun()