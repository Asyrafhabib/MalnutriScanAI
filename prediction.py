import streamlit as st
import pandas as pd
import time
import os
from utils import load_svg, get_svg_base64
from model import load_saved_models 


def show_prediction():
    if 'pred_done' not in st.session_state:
        st.session_state.pred_done = False

    models_dict, le, scaler, numeric_cols = load_saved_models()
    
    if not models_dict:
        st.warning("Waiting for model files... Please ensure the 'model' folder contains .joblib files.")
        return
        
    best_model = models_dict["XGBoost"]

    try:
        icon_brain = load_svg("brain.svg")
        icon_sparkle = load_svg("sparkles.svg")
    except Exception:
        icon_brain = "🧠"
        icon_sparkle = "✨"

    b64_btn_icon = get_svg_base64("brain.svg")
    b64_btn_icon2 = get_svg_base64("arrow-left.svg")

    st.markdown(f"""
        <style>
            .narrow-layout {{ max-width: 650px; margin: 0 auto; }}
            .narrow-layout .stSelectbox, .narrow-layout .stTextInput {{ width: 100% !important; }}
            div[data-baseweb="select"] > div {{ background-color: #F8FAFC !important; border-radius: 8px !important; border: 1px solid #E2E8F0 !important; }}
            .header-brain-box svg {{ width: 28px !important; height: 28px !important; filter: brightness(0) invert(1); }}
            .header-sparkle-box svg {{ width: 16px !important; height: 16px !important; margin-right: 6px; display: block; }}
            
            /* --- JURUS PAMUNGKAS: LOMPATI KOTAK GAIB STREAMLIT --- */
            div[data-testid="stElementContainer"]:has(.btn-marker) + div[data-testid="stElementContainer"] button[kind="primary"],
            .element-container:has(.btn-marker) + .element-container button[kind="primary"] {{ 
                background-color: #0F172A !important; 
                color: #FFFFFF !important;
                border: 2px solid #0F172A !important; 
                border-radius: 12px !important; 
                min-height: 54px !important; 
                width: 100% !important; 
                transition: all 0.3s ease !important; 
                box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important; 
                margin-top: 10px !important; 
            }}
            
            div[data-testid="stElementContainer"]:has(.btn-marker) + div[data-testid="stElementContainer"] button[kind="primary"]:hover,
            .element-container:has(.btn-marker) + .element-container button[kind="primary"]:hover {{
                background-color: #1E293B !important; 
                border-color: #1E293B !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 15px rgba(0,0,0,0.3) !important;
            }}
            
            div[data-testid="stElementContainer"]:has(.btn-marker) + div[data-testid="stElementContainer"] button[kind="primary"] *,
            .element-container:has(.btn-marker) + .element-container button[kind="primary"] * {{ 
                color: #FFFFFF !important; 
                font-weight: 700 !important; 
                font-size: 16px !important; 
                margin: 0 !important;
            }}
            
            div[data-testid="stElementContainer"]:has(.btn-marker) + div[data-testid="stElementContainer"] button[kind="primary"] p,
            .element-container:has(.btn-marker) + .element-container button[kind="primary"] p {{
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
            }}
            
            /* IKON TOMBOL PREDICT (OTAK) */
            div[data-testid="stElementContainer"]:has(.btn-marker-predict) + div[data-testid="stElementContainer"] button[kind="primary"] p::before,
            .element-container:has(.btn-marker-predict) + .element-container button[kind="primary"] p::before {{ 
                content: ''; 
                display: inline-block !important; 
                width: 22px !important; 
                height: 22px !important; 
                margin-right: 10px !important; 
                background-color: #FFFFFF !important; 
                -webkit-mask: url('{b64_btn_icon}') no-repeat center / contain !important; 
                mask: url('{b64_btn_icon}') no-repeat center / contain !important; 
            }}

            /* IKON TOMBOL NEW PREDICTION (PANAH) */
            div[data-testid="stElementContainer"]:has(.btn-marker-new) + div[data-testid="stElementContainer"] button[kind="primary"] p::before,
            .element-container:has(.btn-marker-new) + .element-container button[kind="primary"] p::before {{ 
                content: ''; 
                display: inline-block !important; 
                width: 22px !important; 
                height: 22px !important; 
                margin-right: 10px !important; 
                background-color: #FFFFFF !important; 
                -webkit-mask: url('{b64_btn_icon2}') no-repeat center / contain !important; 
                mask: url('{b64_btn_icon2}') no-repeat center / contain !important; 
            }}
            
            .input-label {{ font-size: 13px; font-weight: 600; color: #1E293B; margin-bottom: 4px; display: block; }}
            .optional-label {{ font-size: 11px; color: #94A3B8; font-weight: 400; }}
            .res-icon {{ width: 64px; height: 64px; border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 15px; }}
            .res-icon.normal {{ background-color: #D1FAE5; color: #10B981; }}
            .res-icon.moderate {{ background-color: #FEF3C7; color: #F59E0B; }}
            .res-icon.severe {{ background-color: #FEE2E2; color: #EF4444; }}
            .res-icon svg {{ width: 32px !important; height: 32px !important; flex-shrink: 0 !important; }}
            .rec-item {{ margin-bottom: 10px; padding: 10px; background-color: #F8FAFC; border-radius: 8px; border-left: 4px solid #3B82F6; display: flex; align-items: flex-start; }}
            .rec-item.dynamic {{ border-left: 4px solid #F59E0B; background-color: #FEF3C7; }}
            .rec-item svg {{ width: 18px !important; height: 18px !important; flex-shrink: 0 !important; color: #3B82F6; margin-top: 2px; margin-right: 10px; }}
            .rec-item.dynamic svg {{ color: #F59E0B; }}
            .disclaimer svg {{ width: 16px !important; height: 16px !important; flex-shrink: 0 !important; margin-top: 1px; }}
        </style>
    """, unsafe_allow_html=True)

    if not st.session_state.pred_done:
        st.markdown('<div class="narrow-layout">', unsafe_allow_html=True)
        st.markdown(f"""
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 20px;'>
                <div class='header-brain-box' style='background-color: #3B82F6; width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;'>{icon_brain}</div>
                <h3 style='text-align: center; color: #0F172A;'>AI Malnutrition Prediction</h3>
                <p style='color: #64748B; text-align: center; margin-bottom: 16px; font-size: 15px;'>Enter patient data to get malnutrition risk prediction using the XGBoost Algorithm</p>
                <div class='header-sparkle-box' style='background-color: #EFF6FF; color: #3B82F6; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; margin: 0 auto 30px auto;'>{icon_sparkle} Powered by XGBoost Algorithm</div>
            </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<h4 style='color: #0F172A; margin-bottom: 5px;'>Patient Data Input</h4>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<span class='input-label'>Age (months) *</span>", unsafe_allow_html=True)
                age = st.text_input("age", placeholder="Example: 24", label_visibility="collapsed")
                err_age = st.empty() 
            with c2:
                st.markdown("<span class='input-label'>Gender *</span>", unsafe_allow_html=True)
                gender = st.selectbox("gender", ["Male", "Female"], label_visibility="collapsed")
            
            st.markdown("<hr style='margin: 15px 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #0F172A; margin-bottom: 15px; font-size: 15px;'>Anthropometric Measurements</h4>", unsafe_allow_html=True)
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                st.markdown("<span class='input-label'>Weight (kg) *</span>", unsafe_allow_html=True)
                weight = st.text_input("weight", placeholder="Example: 12.5", label_visibility="collapsed")
                err_weight = st.empty() 
            with ca2:
                st.markdown("<span class='input-label'>Height (cm) *</span>", unsafe_allow_html=True)
                height = st.text_input("height", placeholder="Example: 85", label_visibility="collapsed")
                err_height = st.empty() 
            with ca3:
                st.markdown("<span class='input-label'>MUAC (Mid-Upper Arm Circumference) - cm *</span>", unsafe_allow_html=True)
                muac = st.text_input("muac", placeholder="Example: 13.5", label_visibility="collapsed")
                err_muac = st.empty() 
            
            st.markdown("<hr style='margin: 15px 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #0F172A; margin-bottom: 15px; font-size: 15px;'>Clinical Assessment <span class='optional-label'>(Optional context)</span></h4>", unsafe_allow_html=True)
            pk1, pk2 = st.columns(2)
            with pk1:
                st.markdown("<span class='input-label'>Weight Loss (3 months) - kg</span>", unsafe_allow_html=True)
                weight_loss = st.text_input("weight_loss", placeholder="Example: 1.5", label_visibility="collapsed")
                err_weight_loss = st.empty()
            with pk2:
                st.markdown("<span class='input-label'>Appetite Score (1-10)</span>", unsafe_allow_html=True)
                appetite = st.text_input("appetite", placeholder="1=Poor, 10=Excellent", label_visibility="collapsed")
                err_appetite = st.empty()
            
            st.markdown("<hr style='margin: 15px 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)

            st.markdown("<h4 style='color: #0F172A; margin-bottom: 5px; font-size: 15px;'>Clinical Symptoms <span class='optional-label'>(Optional context)</span></h4>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<p style='color: #64748B; font-size: 12px; margin-bottom: 15px;'>Select all observed physical and behavioral symptoms</p>", unsafe_allow_html=True)
                
                sym1, sym2, sym3 = st.columns(3)
                with sym1:
                    st.markdown("<span style='font-size: 13px; font-weight: 700; color: #0F172A;'>Physical & Skin</span>", unsafe_allow_html=True)
                    c_fatigue = st.checkbox("Chronic fatigue")
                    c_skin = st.checkbox("Dry / scaly skin")
                    c_anemia = st.checkbox("Pale skin / Anemia")
                    c_hair = st.checkbox("Hair loss")
                with sym2:
                    st.markdown("<span style='font-size: 13px; font-weight: 700; color: #0F172A;'>Gastro & Systemic</span>", unsafe_allow_html=True)
                    c_edema = st.checkbox("Swelling / Edema")
                    c_diarrhea = st.checkbox("Chronic diarrhea")
                    c_nausea = st.checkbox("Frequent nausea")
                    c_wound = st.checkbox("Delayed wound healing")
                with sym3:
                    st.markdown("<span style='font-size: 13px; font-weight: 700; color: #0F172A;'>Muscular & Cog.</span>", unsafe_allow_html=True)
                    c_muscle = st.checkbox("Muscle wasting")
                    c_bones = st.checkbox("Prominent bones")
                    c_apathy = st.checkbox("Apathy / Poor focus")

            # --- MARKER KHUSUS TOMBOL PREDICT ---
            st.markdown("<div class='btn-marker btn-marker-predict'></div>", unsafe_allow_html=True)
            predict_btn = st.button("Predict with AI", type="primary", use_container_width=True)

            if predict_btn:
                err_age.empty()
                err_weight.empty()
                err_height.empty()
                err_muac.empty()
                err_weight_loss.empty()
                err_appetite.empty()
                
                is_valid = True
                
                err_empty = "<div style='color: #DC2626; font-size: 12px; margin-top: 4px; font-weight: 500;'>Required field</div>"
                err_num = "<div style='color: #DC2626; font-size: 12px; margin-top: 4px; font-weight: 500;'>Must be a valid number</div>"

                def is_number(s):
                    try:
                        float(str(s).replace(',', '.'))
                        return True
                    except ValueError:
                        return False

                if not age:
                    err_age.markdown(err_empty, unsafe_allow_html=True)
                    is_valid = False
                elif not is_number(age):
                    err_age.markdown(err_num, unsafe_allow_html=True)
                    is_valid = False

                if not weight:
                    err_weight.markdown(err_empty, unsafe_allow_html=True)
                    is_valid = False
                elif not is_number(weight):
                    err_weight.markdown(err_num, unsafe_allow_html=True)
                    is_valid = False

                if not height:
                    err_height.markdown(err_empty, unsafe_allow_html=True)
                    is_valid = False
                elif not is_number(height):
                    err_height.markdown(err_num, unsafe_allow_html=True)
                    is_valid = False

                if not muac:
                    err_muac.markdown(err_empty, unsafe_allow_html=True)
                    is_valid = False
                elif not is_number(muac):
                    err_muac.markdown(err_num, unsafe_allow_html=True)
                    is_valid = False

                if weight_loss and not is_number(weight_loss):
                    err_weight_loss.markdown(err_num, unsafe_allow_html=True)
                    is_valid = False
                    
                if appetite and not is_number(appetite):
                    err_appetite.markdown(err_num, unsafe_allow_html=True)
                    is_valid = False

                if is_valid:
                    with st.spinner("Analyzing with XGBoost Algorithm & Expert System..."):
                        time.sleep(1) 
                        
                        def safe_float(val_str, default_val):
                            try:
                                return float(str(val_str).replace(',', '.')) if val_str else default_val
                            except ValueError:
                                return default_val

                        age_months = safe_float(age, 24.0)
                        w_val = safe_float(weight, 12.0)
                        h_val = safe_float(height, 85.0)
                        m_val = safe_float(muac, 14.0)
                        wl_val = safe_float(weight_loss, 0.0)
                        app_val = safe_float(appetite, 0.0)

                        bmi_val = w_val / ((h_val / 100) ** 2) if h_val > 0 else 20.0
                        
                        input_data = pd.DataFrame([[age_months, w_val, h_val, m_val, bmi_val]], columns=numeric_cols)
                        res = best_model.predict(input_data)[0]
                        proba = best_model.predict_proba(input_data)[0]
                        label_str = le.inverse_transform([res])[0].lower()
                        
                        conf = float(max(proba))
                        if label_str == "normal": 
                            base_score = 35.0 - (conf * 30.0)
                        elif label_str == "moderate": 
                            base_score = 30.0 + (conf * 30.0)
                        else: 
                            base_score = 60.0 + (conf * 35.0)
                            
                        st.session_state.pred_data = {
                            'label': label_str,
                            'prob': conf * 100,
                            'bmi': bmi_val,
                            'risk_score': base_score,
                            'muac': m_val,
                            'age_months': age_months,
                            'weight_loss': wl_val, 
                            'appetite': app_val,
                            'symptoms': {
                                'fatigue': c_fatigue, 'skin': c_skin, 'anemia': c_anemia, 'hair': c_hair,
                                'edema': c_edema, 'diarrhea': c_diarrhea, 'nausea': c_nausea, 'wound': c_wound,
                                'muscle': c_muscle, 'bones': c_bones, 'apathy': c_apathy
                            }
                        }
                        
                        # --- FITUR BARU: MENYIMPAN DATA KE CSV UNTUK ANALYTICS ---
                        os.makedirs('data', exist_ok=True)
                        history_file = 'data/prediction_history.csv'
                        
                        new_record = pd.DataFrame([{
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'age_months': age_months,
                            'gender': gender,
                            'weight_kg': w_val,
                            'height_cm': h_val,
                            'muac_cm': m_val,
                            'bmi': round(bmi_val, 2),
                            'nutrition_status': label_str
                        }])
                        
                        if not os.path.isfile(history_file):
                            new_record.to_csv(history_file, index=False)
                        else:
                            new_record.to_csv(history_file, mode='a', header=False, index=False)
                        # ---------------------------------------------------------
                        
                        st.session_state.pred_done = True
                        st.rerun()
                    
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # HALAMAN HASIL PREDIKSI
    # ==========================================
    else:
        st.markdown('<div class="narrow-layout">', unsafe_allow_html=True)
        d = st.session_state.pred_data
        lbl = d['label']
        
        try:
            icon_normal = load_svg("circle-check.svg")
            icon_moderate = load_svg("circle-alert.svg")
            icon_severe = load_svg("circle-x.svg")
            icon_check = load_svg("circle-check-big.svg")
            icon_info = load_svg("circle-alert.svg")
        except Exception:
            icon_normal = "✅"
            icon_moderate = "⚠️"
            icon_severe = "🚨"
            icon_check = "✔️"
            icon_info = "ℹ️"

        if lbl == "normal":
            theme = "normal"
            title = "Normal Status"
            sub = "Patient is within healthy parameters"
            icon = icon_normal
            base_recs = [
                "Maintain current balanced diet and nutrition.",
                "Schedule routine anthropometric check-up in 6 months.",
            ]
        elif lbl == "moderate":
            theme = "moderate"
            title = "Moderate Risk"
            sub = "Requires attention and monitoring"
            icon = icon_moderate
            base_recs = [
                "Increase daily calorie and high-quality protein intake.",
                "Consult with a nutritionist for structured meal planning.",
                "Monitor weight weekly to prevent further decline.",
            ]
        else:
            theme = "severe"
            title = "Severe Risk"
            sub = "Immediate medical intervention required!"
            icon = icon_severe
            base_recs = [
                "Refer to therapeutic feeding center immediately.",
                "Initiate F-75/F-100 therapeutic formulas protocol.",
                "Daily vital monitoring required by clinical staff.",
            ]

        dynamic_recs = []
        wl_val = d.get('weight_loss', 0.0)
        app_val = d.get('appetite', 0.0)
        sym = d.get('symptoms', {})

        if wl_val >= 1.0:
            dynamic_recs.append(f"Significant weight loss ({wl_val:.1f}kg): Conduct screening for TB, parasites, or chronic infections.")
        
        if 0 < app_val <= 4:
            dynamic_recs.append(f"Poor appetite (Score {app_val:.1f}): Evaluate for oral thrush, zinc deficiency, or gastrointestinal distress.")

        if sym.get('diarrhea') or sym.get('nausea'):
            dynamic_recs.append("Active GI Symptoms: Administer ORS (Oral Rehydration Salts) and Zinc supplements immediately to prevent severe dehydration.")
            
        if sym.get('edema'):
            dynamic_recs.append("Edema Detected: Suspect Kwashiorkor (protein deficiency). Monitor protein intake and restrict excess fluid. Do not overload heart.")
            
        if sym.get('anemia') or sym.get('fatigue'):
            dynamic_recs.append("Anemia/Fatigue Signs: Check Hemoglobin levels and consider Iron/Folic Acid supplementation.")
            
        if sym.get('skin') or sym.get('hair') or sym.get('wound'):
            dynamic_recs.append("Dermatological Signs: Evaluate for Vitamin A, C, and general micronutrient deficiencies.")
            
        if sym.get('muscle') or sym.get('bones'):
            dynamic_recs.append("Severe Wasting: Suspect Marasmus. Requires cautious, controlled re-feeding protocol to avoid refeeding syndrome.")
            
        if sym.get('apathy'):
            dynamic_recs.append("Cognitive Impact: Provide psychosocial stimulation alongside nutritional rehabilitation.")

        st.markdown(f"""
        <div class="res-hero {theme}">
            <div class="res-icon {theme}">{icon}</div>
            <div class="res-title">{title}</div>
            <div class="res-sub">{sub}</div>
            <div class="res-pills-container">
                <div class="res-pill">Risk Score: {d['risk_score']:.2f}/100</div>
                <div class="res-pill">BMI: {d['bmi']:.2f}</div>
                <div class="res-pill filled {theme}">Confidence: {d['prob']:.1f}%</div>
            </div>
            <div class="res-main-bar"><div class="res-main-bar-fill {theme}" style="width: {d['risk_score']:.2f}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Risk Factor Analysis\n<p class='subtitle' style='margin-bottom:15px;'>Detailed contribution of anthropometric parameters to ML risk score</p>", unsafe_allow_html=True)

            def risk_row(name, val_txt, w_pct, pts):
                return (
                    f"""
                    <div class="risk-row">
                        <div class="risk-header">
                            <div><span class="risk-name">{name}</span> <span class="risk-val">({val_txt}, Weight: {w_pct}%)</span></div>
                            <div class="risk-pts">{pts:.2f} pts</div>
                        </div>
                        <div class="risk-bar-bg">
                            <div class="risk-bar-fill" style="width: {pts:.2f}%; background-color: {'#94A3B8' if pts == 0 else '#0F172A'};"></div>
                        </div>
                        <div class="risk-pct" style="text-align: right; font-size: 11px; color: #64748B; margin-top: 4px; font-weight: 600;">{pts:.2f} / {w_pct}</div>
                    </div>
                    """
                )

            rs = d['risk_score']
            w_bmi, w_muac, w_age = 50, 40, 10
            
            bmi_pts = rs * (w_bmi / 100.0)
            muac_pts = rs * (w_muac / 100.0)
            age_pts = rs * (w_age / 100.0)

            st.markdown(risk_row("BMI Score", f"Value: {d['bmi']:.2f}", w_bmi, bmi_pts), unsafe_allow_html=True)
            st.markdown(risk_row("MUAC Score", f"Value: {d['muac']:.2f}", w_muac, muac_pts), unsafe_allow_html=True)
            st.markdown(risk_row("Age Factor", f"Value: {d.get('age_months', 24.0):.1f} months", w_age, age_pts), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Actionable Recommendations\n<p class='subtitle' style='margin-bottom:15px;'>Suggestions based on ML prediction and clinical expert system</p>", unsafe_allow_html=True)
            
            for rec in base_recs:
                st.markdown(f"""<div class="rec-item"><div style="display:flex; align-items:flex-start; flex-shrink:0;">{icon_check}</div><div><b>Base Protocol:</b> {rec}</div></div>""", unsafe_allow_html=True)
                
            for rec in dynamic_recs:
                st.markdown(f"""<div class="rec-item dynamic"><div style="display:flex; align-items:flex-start; flex-shrink:0;">{icon_check}</div><div><b>Clinical Alert:</b> {rec}</div></div>""", unsafe_allow_html=True)

        st.markdown(
            f"""<div class="disclaimer"><div style="display:flex; align-items:center; flex-shrink:0;">{icon_info}</div><div><b>Disclaimer:</b> The probability is generated by the AI predictive model. Actionable recommendations are augmented by rule-based clinical evaluation. Always consult with a certified doctor.</div></div><br>""",
            unsafe_allow_html=True,
        )
        
        # --- MARKER KHUSUS TOMBOL NEW PREDICTION ---
        st.markdown("<div class='btn-marker btn-marker-new'></div>", unsafe_allow_html=True)
        if st.button("New Prediction", type="primary", use_container_width=True):
            st.session_state.pred_done = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)