import streamlit as st
import pandas as pd
from utils import load_svg, get_svg_base64

def show_dataset():
    # ==========================================
    # 1. LOAD & FORMAT DATASET
    # ==========================================
    try:
        # Membaca file dataset asli
        df = pd.read_csv("malnutrition_data .csv") 
        
        # Tambahkan kolom ID di awal jika belum ada
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
            
        # Format angka agar rapi (maksimal 2 desimal)
        df = df.round(2)
        
        # KAMUS PEMETAAN HEADER (Merubah nama mentah CSV menjadi nama rapi)
        header_mapping = {
            'age_months': 'Age',
            'weight_kg': 'Weight (kg)',
            'height_cm': 'Height (cm)',
            'bmi': 'BMI',
            'muac_cm': 'MUAC (cm)',
            'nutrition_status': 'Status'
        }
        # Terapkan perubahan nama kolom ke DataFrame
        df = df.rename(columns=header_mapping)
        
        # Menyiapkan data untuk fitur Export CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        total_records = len(df)
        total_features = len(df.columns)
        data_loaded = True
    except Exception as e:
        data_loaded = False
        total_records = 0
        total_features = 0
        csv_data = None
        error_msg = str(e)

    # ==========================================
    # 2. SISTEM PEMBERSIH BASE64 UNTUK SEMUA IKON
    # ==========================================
    def prep_b64(filename):
        try:
            b = get_svg_base64(filename).replace("\n", "").replace("\r", "").strip()
            return b if "data:image" in b else f"data:image/svg+xml;base64,{b}"
        except:
            return ""

    final_dl_b64 = prep_b64("download.svg")
    final_search_b64 = prep_b64("search.svg")
    final_filter_b64 = prep_b64("funnel.svg")

    # ==========================================
    # 3. CSS KHUSUS WIDGET STREAMLIT (Sihir UI)
    # ==========================================
    st.markdown(f"""
        <style>
            /* --- 1. Export Button --- */
            [data-testid="stDownloadButton"] button {{
                background-color: #000000 !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 8px !important;
                height: 42px !important;
                width: 100% !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
            }}
            [data-testid="stDownloadButton"] button p {{
                color: #FFFFFF !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }}
            [data-testid="stDownloadButton"] button:hover {{
                background-color: #262626 !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
            }}
            [data-testid="stDownloadButton"] button p::before {{
                content: '';
                display: inline-block;
                width: 18px;
                height: 18px;
                margin-right: 8px;
                background-color: #FFFFFF !important;
                -webkit-mask: url('{final_dl_b64}') no-repeat center / contain;
                mask: url('{final_dl_b64}') no-repeat center / contain;
            }}

            /* --- 2. Custom Search Bar --- */
            [data-testid="stTextInput"] > div > div > input {{
                background-color: #F1F5F9 !important;
                border: none !important;
                font-size: 13px !important;
                padding: 10px 15px 10px 35px !important;
                background-image: url('{final_search_b64}') !important;
                background-repeat: no-repeat !important;
                background-position: 12px center !important;
                background-size: 14px !important;
                color: #475569 !important;
                height: 42px !important;
            }}
            
            /* HILANGKAN TULISAN "PRESS ENTER TO APPLY" SECARA TOTAL */
            [data-testid="InputInstructions"], 
            [data-testid="stInputInstructions"] {{
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
            }}

            [data-testid="stTextInput"] > div > div {{
                background-color: #F1F5F9 !important;
                border: none !important;
                border-radius: 8px !important;
                box-shadow: none !important;
            }}

            /* --- 3. Custom Filter Radio Buttons (Menjadi Pills) --- */
            [data-testid="stRadio"] {{
                width: fit-content !important;
                margin-left: auto !important;
                margin-right: 0 !important;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] {{
                display: flex !important;
                flex-direction: row !important;
                gap: 8px !important;
                flex-wrap: nowrap !important;
                justify-content: flex-end !important;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] > label {{
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                border-radius: 8px !important;
                padding: 6px 14px !important;
                cursor: pointer !important;
                margin: 0 !important;
                transition: all 0.2s ease !important;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
                display: none !important;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
                background-color: #0F172A !important;
                border-color: #0F172A !important;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] > label p {{
                font-size: 13px !important;
                color: #64748B !important;
                font-weight: 500 !important;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {{
                color: #FFFFFF !important;
                font-weight: 600 !important;
            }}
            
            /* KEMBALIKAN IKON FILTER KHUSUS DI TOMBOL 'ALL' (URUTAN 1) */
            [data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1) p::before {{
                content: '';
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 6px;
                background-color: #64748B;
                -webkit-mask: url('{final_filter_b64}') no-repeat center / contain;
                mask: url('{final_filter_b64}') no-repeat center / contain;
                vertical-align: middle;
                margin-top: -2px;
            }}
            [data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(1):has(input:checked) p::before {{
                background-color: #FFFFFF !important;
            }}

            /* --- 4. KUNCI PENYERAGAMAN TINGGI DATA CARD --- */
            .data-card {{
                height: 145px !important; /* Paksa semua kartu berukuran seragam agar tidak melar */
                display: flex !important;
                flex-direction: column !important;
                justify-content: space-between !important; 
            }}
            .dc-title svg {{
                width: 16px !important;
                height: 16px !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 4. RENDER UI BAGIAN ATAS
    # ==========================================
    icon_database = load_svg("database.svg")
    icon_columns = load_svg("file-text.svg")

    c_head1, c_head2 = st.columns([4, 1])
    with c_head1:
        st.markdown("<h1>Training Dataset</h1><p class='subtitle'>Explore the Kaggle malnutrition dataset used for model training</p>", unsafe_allow_html=True)
    with c_head2:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if csv_data is not None:
            st.download_button(
                label="Export CSV",  
                data=csv_data,
                file_name="MalnutriScan_Dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="data-card"><div class="dc-title" style="display: flex; align-items: center; gap: 8px;"><div style="width:16px; height:16px; display:flex; align-items: center;">{icon_database}</div> <span>Total Records</span></div><div class="dc-val">{total_records:,}</div><div class="dc-sub">Training samples</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="data-card"><div class="dc-title" style="display: flex; align-items: center; gap: 8px;"><div style="width:16px; height:16px; display:flex; align-items: center;">{icon_columns}</div> <span>Features</span></div><div class="dc-val">{total_features}</div><div class="dc-sub">Data columns</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="data-card"><div class="dc-title">Source</div><div class="dc-val"><a href="https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset" target="_blank" style="text-decoration: none;"><span class="kaggle-pill" style="cursor: pointer;">Kaggle</span></a></div><div class="dc-sub">Nutrition Dataset</div></div>""", unsafe_allow_html=True)
    with s4:
        st.markdown("""<div class="data-card"><div class="dc-title">Last Updated</div><div class="dc-val" style="font-size: 22px; font-weight: 700;">Feb 2026</div><div class="dc-sub">Current version</div></div>""", unsafe_allow_html=True)

    # Memberikan Jarak (Spacing) antara Data Cards dan Container Data Explorer
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # 5. PANEL DATA EXPLORER & FILTERING LOGIC
    # ==========================================
    with st.container(border=True):
        st.markdown("#### Data Explorer\n<p class='subtitle' style='margin-bottom: 10px;'>Search and filter the dataset</p>", unsafe_allow_html=True)
        
        if data_loaded:
            c_search, c_filter = st.columns([1.8, 1])
            with c_search:
                search_query = st.text_input("Search", placeholder="Search by ID or Age...", label_visibility="collapsed")
            with c_filter:
                filter_val = st.radio("Filter", ["All", "Normal", "Moderate", "Severe"], horizontal=True, label_visibility="collapsed")
            
            # --- LOGIKA PENYARINGAN DATA ---
            df_display = df.copy()
            
            if search_query:
                mask = df_display.astype(str).apply(lambda x: x.str.contains(search_query, case=False, na=False)).any(axis=1)
                df_display = df_display[mask]
                
            if filter_val and filter_val != "All":
                mask = df_display['Status'].astype(str).str.contains(filter_val, case=False, na=False)
                df_display = df_display[mask]
                
            total_filtered = len(df_display)
            display_limit = min(10, total_filtered)
            
            if total_filtered == total_records:
                st.markdown(f"<div style='color: #94A3B8; font-size: 12px; margin-bottom: 10px; margin-top: -5px;'>Showing {display_limit} of {total_records:,} records</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color: #94A3B8; font-size: 12px; margin-bottom: 10px; margin-top: -5px;'>Showing {display_limit} of {total_filtered:,} filtered records (from {total_records:,} total)</div>", unsafe_allow_html=True)
                
            # --- GENERATE TABEL HTML HASIL FILTER ---
            headers_html = "".join([f"<th>{col}</th>" for col in df_display.columns])
            rows_html = ""
            
            if total_filtered == 0:
                rows_html = f"<tr><td colspan='{total_features}' style='text-align:center; padding:30px; color:#94A3B8;'>No records found matching your criteria.</td></tr>"
            else:
                for _, row in df_display.head(display_limit).iterrows():
                    row_html = "<tr>"
                    for col in df_display.columns:
                        val = row[col]
                        if str(col).lower() in ['status', 'label', 'malnutrition_status', 'risk', 'class']:
                            val_str = str(val).capitalize()
                            pill_class = "p-normal"
                            if "moderate" in val_str.lower(): pill_class = "p-moderate"
                            elif "severe" in val_str.lower() or "sam" in val_str.lower(): pill_class = "p-severe"
                            row_html += f'<td><span class="s-pill {pill_class}">{val_str}</span></td>'
                        else:
                            row_html += f'<td>{val}</td>'
                    row_html += "</tr>"
                    rows_html += row_html
            
            st.markdown(f"""
            <div class="custom-table-container">
                <table class="custom-table">
                    <thead><tr>{headers_html}</tr></thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error(f"Gagal memuat dataset: {error_msg}")

    # ==========================================
    # 6. DATA DICTIONARY
    # ==========================================
    with st.container(border=True):
        st.markdown("#### Data Dictionary\n<p class='subtitle'>Description of dataset features</p>", unsafe_allow_html=True)
        col_d1, col_spacer, col_d2 = st.columns([1, 0.05, 1])

        def dict_item(title, desc, badge):
            return f"""<div class="dict-item"><div><div class="dict-title">{title}</div><div class="dict-desc">{desc}</div></div><div class="badge-type">{badge}</div></div>"""

        with col_d1:
            st.markdown(dict_item("Age", "Patient's age in months", "Numeric"), unsafe_allow_html=True)
            st.markdown(dict_item("Weight (kg)", "Body weight in kilograms", "Numeric"), unsafe_allow_html=True)
            st.markdown(dict_item("Height (cm)", "Body height in centimeters", "Numeric"), unsafe_allow_html=True)
        with col_d2:
            st.markdown(dict_item("BMI", "Calculated Body Mass Index", "Numeric"), unsafe_allow_html=True)
            st.markdown(dict_item("MUAC (cm)", "Mid-Upper Arm Circumference", "Numeric"), unsafe_allow_html=True)
            st.markdown(dict_item("Status", "Final malnutrition classification", "Categorical"), unsafe_allow_html=True)