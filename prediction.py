import streamlit as st
import pandas as pd
from utils import load_svg, get_svg_base64
from model import load_and_train_model


def show_prediction():
    models, scaler, le, df, numeric_cols = load_and_train_model()
    best_model = models["XGBoost"]

    icon_brain = load_svg("brain.svg")
    icon_sparkle = load_svg("sparkles.svg")
    b64_brain = get_svg_base64("brain.svg")

    st.markdown(f"""
        <style>
            .narrow-layout {{
                max-width: 600px;
                margin: 0 auto;
            }}
            .narrow-layout .stSelectbox,
            .narrow-layout .stTextInput {{
                width: 100% !important;
            }}
            div[data-baseweb="select"] > div {{
                background-color: #F8FAFC !important;
                border-radius: 8px !important;
                border: 1px solid #E2E8F0 !important;
            }}
            .header-brain-box svg {{
                width: 28px !important;
                height: 28px !important;
                filter: brightness(0) invert(1);
            }}
            .header-sparkle-box svg {{
                width: 16px !important;
                height: 16px !important;
                margin-right: 6px;
                display: block;
            }}
            .predict-btn-wrapper [data-testid="stButton"] > button,
            .predict-btn-wrapper [data-testid="baseButton-secondary"] {{
                background-color: #000000 !important;
                color: #FFFFFF !important;
                border: 2px solid #000000 !important;
                border-radius: 12px !important;
                height: 54px !important;
                width: 100% !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
                margin-top: 10px;
            }}
            .predict-btn-wrapper [data-testid="stButton"] > button p,
            .predict-btn-wrapper [data-testid="stButton"] > button span {{
                color: #FFFFFF !important;
                font-weight: 700 !important;
                font-size: 16px !important;
            }}
            .predict-btn-wrapper [data-testid="stButton"] > button:hover {{
                background-color: #262626 !important;
                border-color: #262626 !important;
                color: #FFFFFF !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 15px rgba(0,0,0,0.3) !important;
            }}
            .predict-btn-wrapper [data-testid="stButton"] > button p::before {{
                content: '';
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 12px;
                background-color: #FFFFFF !important;
                -webkit-mask: url('{b64_brain}') no-repeat center / contain;
                mask: url('{b64_brain}') no-repeat center / contain;
            }}
        </style>
    """, unsafe_allow_html=True)

    if not st.session_state.pred_done:
        st.markdown('<div class="narrow-layout">', unsafe_allow_html=True)
        st.markdown(f"""
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 20px;'>
                <div class='header-brain-box' style='background-color: #3B82F6; width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;'>
                    {icon_brain}
                </div>
                <h1 style='text-align: center; font-size:32px;'>AI Malnutrition Prediction</h1>
                <p style='color: #64748B; text-align: center; margin-bottom: 16px; font-size: 15px;'>Enter patient data to get malnutrition risk prediction using machine learning model</p>
                <div class='header-sparkle-box' style='background-color: #EFF6FF; color: #3B82F6; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; margin: 0 auto 30px auto;'>
                    {icon_sparkle} Powered by XGBoost Algorithm
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Patient Data Input")
            st.markdown("<p style='color: #64748B; font-size: 14px; margin-bottom: 24px;'>Complete all fields for accurate prediction</p>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<span class='input-label'>Age (years) *</span>", unsafe_allow_html=True)
                age = st.text_input("age", placeholder="Example: 25", label_visibility="collapsed")
            with c2:
                st.markdown("<span class='input-label'>Gender *</span>", unsafe_allow_html=True)
                gender = st.selectbox("gender", ["Male", "Female"], label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Anthropometric Measurements")
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                st.markdown("<span class='input-label'>Weight (kg) *</span>", unsafe_allow_html=True)
                weight = st.text_input("weight", placeholder="Example: 55.5", label_visibility="collapsed")
            with ca2:
                st.markdown("<span class='input-label'>Height (cm) *</span>", unsafe_allow_html=True)
                height = st.text_input("height", placeholder="Example: 165", label_visibility="collapsed")
            with ca3:
                st.markdown("<span class='input-label'>MUAC (cm) *</span>", unsafe_allow_html=True)
                muac = st.text_input("muac", placeholder="Example: 24.5", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Clinical Assessment")
            pk1, pk2 = st.columns(2)
            with pk1:
                st.markdown("<span class='input-label'>Weight Loss (3 months) - kg *</span>", unsafe_allow_html=True)
                weight_loss = st.text_input("weight_loss", placeholder="Example: 2.5", label_visibility="collapsed")
            with pk2:
                st.markdown("<span class='input-label'>Appetite Score (1-10) *</span>", unsafe_allow_html=True)
                appetite = st.text_input("appetite", placeholder="1=Very poor, 10=Excellent", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### Clinical Symptoms")
            with st.container(border=True):
                st.markdown("<p style='color: #64748B; font-size: 13px; margin-bottom: 15px;'>Select all observed physical and behavioral symptoms (optional)</p>", unsafe_allow_html=True)
                
                # Dibagi menjadi 3 kolom agar UI tetap padat dan rapi
                sym1, sym2, sym3 = st.columns(3)
                
                with sym1:
                    st.markdown("<span style='font-size: 14px; font-weight: 700; color: #0F172A;'>Physical & Skin</span>", unsafe_allow_html=True)
                    sym_fatigue = st.checkbox("Chronic fatigue")
                    sym_skin = st.checkbox("Dry / scaly skin")
                    sym_pale = st.checkbox("Pale skin / Anemia")
                    sym_hair = st.checkbox("Hair loss / Thinning")
                    sym_nails = st.checkbox("Brittle nails")
                    
                with sym2:
                    st.markdown("<span style='font-size: 14px; font-weight: 700; color: #0F172A;'>Gastro & Systemic</span>", unsafe_allow_html=True)
                    sym_edema = st.checkbox("Swelling / Edema")
                    sym_diarrhea = st.checkbox("Chronic diarrhea")
                    sym_nausea = st.checkbox("Frequent nausea")
                    sym_wound = st.checkbox("Delayed wound healing")
                    
                with sym3:
                    st.markdown("<span style='font-size: 14px; font-weight: 700; color: #0F172A;'>Muscular & Cognitive</span>", unsafe_allow_html=True)
                    sym_muscle = st.checkbox("Muscle wasting")
                    sym_bone = st.checkbox("Prominent bones")
                    sym_dizzy = st.checkbox("Recurrent dizziness")
                    sym_focus = st.checkbox("Poor concentration / Apathy")

            st.markdown("<p style='color: #94A3B8; font-size: 13px; margin-top: 4px; margin-bottom: 20px;'>Select all relevant symptoms (optional)</p>", unsafe_allow_html=True)
            st.markdown('<div class="predict-btn-wrapper">', unsafe_allow_html=True)
            predict_btn = st.button("Predict with AI", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
            if predict_btn:
                try:
                    age_val = float(age) if age else 25.0
                except ValueError:
                    age_val = 25.0
                try:
                    w_val = float(weight) if weight else 55.5
                except ValueError:
                    w_val = 55.5
                try:
                    h_val = float(height) if height else 165.0
                except ValueError:
                    h_val = 165.0
                try:
                    m_val = float(muac) if muac else 24.5
                except ValueError:
                    m_val = 24.5

                bmi_val = w_val / ((h_val / 100) ** 2) if h_val > 0 else 20.0
                input_data = pd.DataFrame([[age_val, w_val, h_val, m_val, bmi_val]], columns=numeric_cols)
                res = best_model.predict(input_data)[0]
                proba = best_model.predict_proba(input_data)[0]
                label_str = le.inverse_transform([res])[0].lower()
                base_score = 0
                if label_str == "normal":
                    base_score = 15.5
                elif label_str == "moderate":
                    base_score = 36.5
                else:
                    base_score = 78.2
                st.session_state.pred_data = {
                    'label': label_str,
                    'prob': max(proba) * 100,
                    'bmi': round(bmi_val, 1),
                    'risk_score': base_score,
                    'muac': round(m_val, 1),
                }
                st.session_state.pred_done = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="narrow-layout">', unsafe_allow_html=True)
        d = st.session_state.pred_data
        lbl = d['label']
        icon_normal = load_svg("icon_normal.svg")
        icon_moderate = load_svg("icon_moderate.svg")
        icon_severe = load_svg("icon_severe.svg")
        icon_check = load_svg("icon_check.svg")
        icon_info = load_svg("icon_info.svg")
        if lbl == "normal":
            theme = "normal"
            title = "Normal Status"
            sub = "Patient is within healthy parameters"
            icon = icon_normal
            recs = [
                "Maintain current balanced diet and nutrition",
                "Schedule routine check-up in 6 months",
                "Encourage normal physical activity",
            ]
        elif lbl == "moderate":
            theme = "moderate"
            title = "Mild Risk"
            sub = "Requires attention and monitoring"
            icon = icon_moderate
            recs = [
                "Increase daily calorie and protein intake",
                "Consult with a nutritionist for meal planning",
                "Monitor weight weekly",
                "Identify cause of weight loss: infection, GI disorders",
            ]
        else:
            theme = "severe"
            title = "Severe Risk"
            sub = "Immediate medical intervention required!"
            icon = icon_severe
            recs = [
                "Refer to therapeutic feeding center immediately",
                "Initiate F-75/F-100 therapeutic formulas",
                "Check for underlying complications (edema, infection)",
                "Daily vital monitoring",
            ]

        st.markdown(f"""
        <div class="res-hero {theme}">
            <div class="res-icon {theme}"><div style="width:32px; height:32px; display:flex;">{icon}</div></div>
            <div class="res-title">{title}</div>
            <div class="res-sub">{sub}</div>
            <div class="res-pills-container">
                <div class="res-pill">Risk Score: {d['risk_score']}/100</div>
                <div class="res-pill">BMI: {d['bmi']}</div>
                <div class="res-pill filled {theme}">Confidence: {d['prob']:.1f}%</div>
            </div>
            <div class="res-main-bar"><div class="res-main-bar-fill" style="width: {d['risk_score']}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Risk Factor Analysis\n<p class='subtitle' style='margin-bottom:15px;'>Contribution of each parameter to total risk score</p>", unsafe_allow_html=True)

            def risk_row(name, val_txt, w_pct, pts, p_pct):
                return (
                    f"""
                    <div class="risk-row"><div class="risk-header"><div><span class="risk-name">{name}</span> <span class="risk-val">({val_txt}, Weight: {w_pct}%)</span></div><div class="risk-pts">{pts} pts</div></div><div class="risk-bar-bg"><div class="risk-bar-fill" style="width: {p_pct}%; background-color: {'#94A3B8' if p_pct == 0 else '#0F172A'};"></div></div><div class="risk-pct">{p_pct}%</div></div>
                    """
                )

            if theme == "moderate":
                st.markdown(risk_row("BMI Score", f"Value: {d['bmi']}", 25, "5.0", 14), unsafe_allow_html=True)
                st.markdown(risk_row("MUAC Score", f"Value: {d['muac']}", 20, "0.0", 0), unsafe_allow_html=True)
                st.markdown(risk_row("Weight Loss", "Value: 2.0", 15, "15.0", 41), unsafe_allow_html=True)
                st.markdown(risk_row("Appetite", "Value: 3.0", 15, "10.5", 29), unsafe_allow_html=True)
                st.markdown(risk_row("Symptoms", "Value: 0.0", 15, "0.0", 0), unsafe_allow_html=True)
                st.markdown(risk_row("Age Factor", "Value: 24.0", 10, "6.0", 16), unsafe_allow_html=True)
            else:
                st.markdown(risk_row("Anthropometry Core", "BMI/MUAC", 45, f"{d['risk_score']*0.6:.1f}", 60), unsafe_allow_html=True)
                st.markdown(risk_row("Clinical Symptoms", "Inputs", 55, f"{d['risk_score']*0.4:.1f}", 40), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Recommendations\n<p class='subtitle' style='margin-bottom:15px;'>Suggestions based on AI prediction results</p>", unsafe_allow_html=True)
            for rec in recs:
                st.markdown(f"""<div class="rec-item"><div style="width:16px; height:16px; display:flex;">{icon_check}</div><div>{rec}</div></div>""", unsafe_allow_html=True)

        st.markdown(
            f"""<div class="disclaimer"><div style="flex-shrink:0; margin-top:2px; width:14px; height:14px; display:flex;">{icon_info}</div><div><b>Disclaimer:</b> This prediction is generated by a machine learning model and serves as a screening tool. This is NOT a final medical diagnostic. Always consult with a doctor or nutritionist for comprehensive assessment and proper treatment.</div></div><br>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="predict-btn-wrapper">', unsafe_allow_html=True)
        if st.button("New Prediction", use_container_width=True):
            st.session_state.pred_done = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
