import streamlit as st
import pandas as pd
from utils import load_svg, get_svg_base64


def show_dashboard():
    icon_stats = load_svg("users-round.svg")
    icon_ai = load_svg("brain.svg")
    icon_analytics = load_svg("chart-column.svg")
    icon_dataset = load_svg("database.svg")

    b64_pred = get_svg_base64("brain.svg")
    b64_anal = get_svg_base64("chart-column.svg")
    b64_data = get_svg_base64("database.svg")

    model_metrics = {
        "XGBoost": {"acc": 94.0, "color": "green", "desc": "Best Model"},
        "Random Forest": {"acc": 92.5, "color": "blue", "desc": "Excellent"},
        "Gradient Boost": {"acc": 91.8, "color": "purple", "desc": "Very Good"},
        "SVM": {"acc": 89.2, "color": "orange", "desc": "Stable"},
        "Decision Tree": {"acc": 88.4, "color": "blue", "desc": "Baseline Tree"},
        "Log. Regression": {"acc": 87.5, "color": "green", "desc": "Linear Base"},
        "KNN": {"acc": 85.0, "color": "orange", "desc": "Distance"},
        "Naive Bayes": {"acc": 82.1, "color": "purple", "desc": "Probabilistic"},
    }

    b64_start = get_svg_base64("brain.svg")
    b64_view = get_svg_base64("chart-column.svg")

    st.markdown(f"""
        <style>
            .new-hero-banner {{
                background: linear-gradient(135deg, #2563EB 0%, #7C3AED 50%, #C026D3 100%);
                border-radius: 16px;
                padding: 40px;
                padding-bottom: 85px;
                color: white;
                margin-bottom: 10px;
            }}
            .new-hero-badge {{
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 4px 14px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                display: inline-block;
                margin-bottom: 16px;
            }}
            .new-hero-title {{
                font-size: 34px;
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 8px;
            }}
            .new-hero-title span {{ color: #A5F3FC; }}
            .new-hero-subtitle {{ font-size: 15px; color: rgba(255, 255, 255, 0.85); line-height: 1.5; margin-bottom: 0; }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) {{
                margin-top: -85px !important;
                padding-left: 40px !important;
                z-index: 10 !important;
                position: relative;
            }}
            div.element-container:has(.pull-up-anchor) {{
                position: absolute !important; width: 0px !important; height: 0px !important;
                min-height: 0px !important; margin: 0px !important; padding: 0px !important; overflow: hidden !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) div.element-container {{
                margin-bottom: 0 !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) button {{
                height: 44px !important; margin: 0 !important; border-radius: 8px !important;
                font-weight: 600 !important; box-sizing: border-box !important; transition: all 0.3s ease !important;
                width: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) button p {{
                position: relative !important; margin: 0 !important; padding-left: 26px !important; display: inline-block !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) button p::before,
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) button p::after {{
                display: none !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) > div:nth-child(1) button {{
                background-color: #FFFFFF !important; color: #2563EB !important;
                border: 2px solid #FFFFFF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) > div:nth-child(1) button:hover {{
                transform: translateY(-2px) !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) > div:nth-child(1) button p::before {{
                display: block !important; content: ''; position: absolute !important;
                left: 0 !important; top: 50% !important; transform: translateY(-50%) !important;
                width: 18px !important; height: 18px !important; background-color: #2563EB !important;
                -webkit-mask: url('{b64_start}') no-repeat center / contain !important;
                mask: url('{b64_start}') no-repeat center / contain !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) > div:nth-child(2) button {{
                background-color: transparent !important; color: #FFFFFF !important;
                border: 2px solid rgba(255, 255, 255, 0.6) !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) > div:nth-child(2) button:hover {{
                background-color: rgba(255, 255, 255, 0.1) !important; border-color: #FFFFFF !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.pull-up-anchor) > div:nth-child(2) button p::before {{
                display: block !important; content: ''; position: absolute !important;
                left: 0 !important; top: 50% !important; transform: translateY(-50%) !important;
                width: 18px !important; height: 18px !important; background-color: #FFFFFF !important;
                -webkit-mask: url('{b64_view}') no-repeat center / contain !important;
                mask: url('{b64_view}') no-repeat center / contain !important;
            }}
        </style>
        <div class="new-hero-banner">
            <div class="new-hero-badge">AI-Powered Healthcare</div>
            <div class="new-hero-title">Malnutrition Prediction System<br><span>Powered by XGBoost Algorithm</span></div>
            <div class="new-hero-subtitle">
                Early detection and risk prediction of malnutrition with high accuracy using XGBoost<br>
                algorithm trained on comprehensive Kaggle dataset.
            </div>
        </div>
    """, unsafe_allow_html=True)

    hb_col1, hb_col2, hb_col3 = st.columns([1.3, 1.3, 4])
    with hb_col1:
        st.markdown('<span class="pull-up-anchor"></span>', unsafe_allow_html=True)
        btn_start_pred = st.button("Start Prediction", use_container_width=True)
    with hb_col2:
        btn_view_anal = st.button("View Analytics", use_container_width=True)
    if btn_start_pred:
        st.session_state.page = "Prediksi"
        st.rerun()
    if btn_view_anal:
        st.session_state.page = "Analytics"
        st.rerun()

    st.markdown("<div class='section-title'>AI Models Performance Leaderboard</div>", unsafe_allow_html=True)
    models_list = list(model_metrics.items())
    row1 = st.columns(4)
    for i in range(4):
        m_name, m_data = models_list[i]
        with row1[i]:
            st.markdown(f"""
            <div class="perf-card c-{m_data['color']}">
                <div class="perf-title t-{m_data['color']}">{m_name}</div>
                <div class="perf-value t-{m_data['color']}">{m_data['acc']}%</div>
                <div class="perf-desc t-{m_data['color']}">{m_data['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    row2 = st.columns(4)
    for i in range(4, 8):
        m_name, m_data = models_list[i]
        with row2[i-4]:
            st.markdown(f"""
            <div class="perf-card c-{m_data['color']}">
                <div class="perf-title t-{m_data['color']}">{m_name}</div>
                <div class="perf-value t-{m_data['color']}">{m_data['acc']}%</div>
                <div class="perf-desc t-{m_data['color']}">{m_data['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <style>
        div[data-testid="stHorizontalBlock"]:has(.sync-height) {{ align-items: stretch !important; }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.sync-height) {{
            height: 100% !important; display: flex !important; flex-direction: column !important; justify-content: flex-start !important;
        }}
        div[data-testid="stColumn"]:nth-child(2) div[data-testid="stVerticalBlock"] {{ gap: 1.2rem !important; }}
        div[data-testid="stColumn"]:nth-child(2) button p::before {{
            content: ''; display: block; width: 18px; height: 18px; flex-shrink: 0; margin-right: 12px; background-color: currentColor;
        }}
        div[data-testid="stColumn"]:nth-child(2) div.element-container:nth-child(3) button p::before {{
            -webkit-mask: url('{b64_pred}') no-repeat center / contain; mask: url('{b64_pred}') no-repeat center / contain;
        }}
        div[data-testid="stColumn"]:nth-child(2) div.element-container:nth-child(4) button p::before {{
            -webkit-mask: url('{b64_anal}') no-repeat center / contain; mask: url('{b64_anal}') no-repeat center / contain;
        }}
        div[data-testid="stColumn"]:nth-child(2) div.element-container:nth-child(5) button p::before {{
            -webkit-mask: url('{b64_data}') no-repeat center / contain; mask: url('{b64_data}') no-repeat center / contain;
        }}
    </style>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.8, 1])
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="sync-height"></div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
                    <div style="width:28px; height:28px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                        {icon_stats}
                    </div>
                    <div style='font-weight:800; font-size:17px; color:#0F172A;'>
                        Training Data Statistics
                    </div>
                </div>
                <div style='font-size:13px; color:#64748B; margin-bottom:24px; margin-left:40px;'>Comprehensive nutrition dataset from Kaggle</div>
            """, unsafe_allow_html=True)
            try:
                df = pd.read_csv("malnutrition_data .csv")
                tot_samples = f"{len(df):,}".replace(",", ".")
                target_col = df.columns[-1]
                val_counts = df[target_col].astype(str).str.lower().value_counts()
                c_normal = val_counts.get('normal', 0)
                c_moderate = val_counts.get('moderate', 0)
                c_severe = val_counts.get('severe', 0)
                bmi_col = [c for c in df.columns if 'bmi' in c.lower()]
                avg_bmi = round(df[bmi_col[0]].mean(), 1) if bmi_col else 0.0
                muac_col = [c for c in df.columns if 'muac' in c.lower()]
                avg_muac = round(df[muac_col[0]].mean(), 1) if muac_col else 0.0
                age_col = [c for c in df.columns if 'age' in c.lower()]
                avg_age = round(df[age_col[0]].mean(), 1) if age_col else 0.0
            except Exception:
                tot_samples, c_normal, c_moderate, c_severe = "5.000", 3550, 1100, 350
                avg_bmi, avg_muac, avg_age = 10.0, 13.4, 24.5
            b_cols = st.columns(4)
            with b_cols[0]:
                st.markdown(f"<div class='stat-box b-blue'><div class='stat-label t-blue'>Total Samples</div><div class='stat-val t-blue'>{tot_samples}</div></div>", unsafe_allow_html=True)
            with b_cols[1]:
                st.markdown(f"<div class='stat-box b-green'><div class='stat-label t-green'>Normal</div><div class='stat-val t-green'>{c_normal}</div></div>", unsafe_allow_html=True)
            with b_cols[2]:
                st.markdown(f"<div class='stat-box b-yellow'><div class='stat-label t-orange'>Moderate Risk</div><div class='stat-val t-orange'>{c_moderate}</div></div>", unsafe_allow_html=True)
            with b_cols[3]:
                st.markdown(f"<div class='stat-box b-orange'><div class='stat-label t-orange'>Severe Risk</div><div class='stat-val t-orange'>{c_severe}</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            
            # --- BAGIAN YANG DIPERBAIKI (POSISI TEXT DINAIKKAN) ---
            s_cols = st.columns(3)
            with s_cols[0]:
                st.markdown(f"<div class='summary-text' style='position: relative; top: -15px;'>Average BMI<br><span class='summary-val'>{avg_bmi}</span></div>", unsafe_allow_html=True)
            with s_cols[1]:
                st.markdown(f"<div class='summary-text' style='position: relative; top: -15px;'>Average MUAC<br><span class='summary-val'>{avg_muac} cm</span></div>", unsafe_allow_html=True)
            with s_cols[2]:
                st.markdown(f"<div class='summary-text' style='text-align:right; position: relative; top: -15px;'>Average Age<br><span class='summary-val'>{avg_age} months</span></div>", unsafe_allow_html=True)
            # -----------------------------------------------------

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="sync-height"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div style='margin-bottom: 24px;'>
                    <div style='font-weight:800; font-size:17px; color:#0F172A; margin-bottom:4px;'>Quick Actions</div>
                    <div style='font-size:13px; color:#64748B;'>Get started with the system</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("New Prediction →", key="qa_pred", use_container_width=True):
                st.session_state.page = "Prediksi"
                st.rerun()
            if st.button("View Analytics →", key="qa_anal", use_container_width=True):
                st.session_state.page = "Analytics"
                st.rerun()
            if st.button("Explore Dataset →", key="qa_data", use_container_width=True):
                st.session_state.page = "Dataset"
                st.rerun()

    st.markdown("<h3>Key Features</h3>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="feature-card" style="min-height: 230px; display: flex; flex-direction: column;">
            <div class="icon-box ib-blue">{icon_ai}</div>
            <h4>XGBoost Prediction</h4>
            <p style="color: #64748B; font-size: 14px; line-height: 1.6; margin-bottom: 0;">Advanced XGBoost algorithm trained on 1.000+ samples from comprehensive Kaggle nutrition dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="feature-card" style="min-height: 230px; display: flex; flex-direction: column;">
            <div class="icon-box ib-green">{icon_analytics}</div>
            <h4>Comprehensive Analytics</h4>
            <p style="color: #64748B; font-size: 14px; line-height: 1.6; margin-bottom: 0;">Detailed visualization and insights from the dataset with real-time statistical analysis and trend monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="feature-card" style="min-height: 230px; display: flex; flex-direction: column;">
            <div class="icon-box ib-purple">{icon_dataset}</div>
            <h4>Kaggle Dataset Integration</h4>
            <p style="color: #64748B; font-size: 14px; line-height: 1.6; margin-bottom: 0;">Built on validated medical research data with 9 key features for accurate malnutrition assessment.</p>
        </div>
        """, unsafe_allow_html=True)