import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import sys
import importlib

# Defensive import to handle Streamlit module-reloading KeyError
if 'Attrition_system' in sys.modules:
    importlib.reload(sys.modules['Attrition_system'])
import Attrition_system as asy

# Extract functions for easier access
load_model = asy.load_model
sample_employee = asy.sample_employee
classify_employee_value = asy.classify_employee_value
recommend_actions = asy.recommend_actions
preprocess_input = asy.preprocess_input
get_attrition_drivers = asy.get_attrition_drivers

# --- Configuration & Styling ---
st.set_page_config(page_title="HR Attrition Decision Support", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Styles - PREMIUM & COMPACT */
    .stCard, [data-testid="stVerticalBlock"] > div > div > [data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #f1f5f9;
    }
    
    .section-header {
        background: linear-gradient(90deg, #f8fafc 0%, #eff6ff 100%);
        padding: 4px 12px;
        border-radius: 6px;
        text-align: left;
        font-weight: 800;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin: 6px 0 4px 0;
        border-left: 3px solid #3b82f6;
    }

    .risk-score {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        margin: 5px 0;
        letter-spacing: -0.02em;
    }
</style>
""", unsafe_allow_html=True)

# --- Feature Mapping (Technical -> Human Readable) ---
FEATURE_MAP = {
    'DistanceFromHome': 'Commute distance',
    'OverTime': 'Overtime frequency',
    'YearsSinceLastPromotion': 'Promotion stagnation',
    'EnvironmentSatisfaction': 'Environment dissatisfaction',
    'JobSatisfaction': 'Low job satisfaction',
    'WorkLifeBalance': 'Work-life balance issues',
    'NumCompaniesWorked': 'Job hopping history',
    'YearsWithCurrManager': 'Manager relationship tenure',
    'MonthlyIncome': 'Income level impact',
    'Age': 'Demographic age factor',
    'YearsAtCompany': 'Company tenure factor',
    'MaritalStatus': 'Marital status factor'
}

# --- Resource Loading ---
@st.cache_resource
def get_model():
    if os.path.exists("attrition_model.pkl"):
        return load_model("attrition_model.pkl")
    return None

@st.cache_data
def get_data():
    if os.path.exists('WA_Fn-UseC_-HR-Employee-Attrition.csv'):
        return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    return None

model = get_model()
df_raw = get_data()

# --- Logic: Analysis Engine ---
def run_strategic_analysis(model, df, threshold, want_leave, value_type, min_risk, max_risk):
    # Call original selector
    sample = sample_employee(model, df, threshold, want_leave, value_type, min_risk, max_risk)
    
    if sample is None:
        return None
    
    idx = sample["index"]
    X_row = sample["X_row"]
    pred = sample["pred"]
    proba = sample["proba"]
    
    raw_employee = df.loc[idx]
    
    # Ground Truth logic
    true_raw = raw_employee['Attrition']
    actual = "Leave" if true_raw == "Yes" or true_raw == 1 else "Stay"
    
    # Driver Calculation (Using rule-based system from user)
    structured_drivers = get_attrition_drivers(X_row.iloc[0])
    
    # Values & Recommendations (Using the new signature)
    score, value_label = classify_employee_value(X_row.iloc[0])
    recs = recommend_actions(X_row.iloc[0], proba, value_label, structured_drivers, threshold)
    
    # Derived Insights
    work_intensity = "High" if (raw_employee.get('OverTime') == 'Yes' or X_row['OverTime'].values[0] == 1) and raw_employee['WorkLifeBalance'] <= 2 else "Medium" if (raw_employee.get('OverTime') == 'Yes' or X_row['OverTime'].values[0] == 1) else "Low"
    
    # Income level compared to JobLevel average
    avg_income = df[df['JobLevel'] == raw_employee['JobLevel']]['MonthlyIncome'].mean()
    income_level = "Above Average" if raw_employee['MonthlyIncome'] > avg_income * 1.1 else "Below Average" if raw_employee['MonthlyIncome'] < avg_income * 0.9 else "Standard"
    
    stability = "Stable" if raw_employee['YearsAtCompany'] > 5 and raw_employee['NumCompaniesWorked'] < 3 else "Moderate" if raw_employee['YearsAtCompany'] > 2 else "Low"

    return {
        "index": idx,
        "raw": raw_employee,
        "X_row": X_row.iloc[0],
        "risk": proba,
        "prediction": "Leave" if pred == 1 else "Stay",
        "actual": actual,
        "value_label": value_label,
        "value_score": score,
        "problems": recs["Problems"],
        "actions": recs["Actions"],
        "structured_drivers": structured_drivers,
        "insights": {
            "Work Intensity": work_intensity,
            "Income Alignment": income_level,
            "Career Stability": stability
        }
    }

# --- Sidebar: Redesigned ---
with st.sidebar:
    st.title("⚙️ HR Strategy Hub")
    st.caption("Nordea Attrition Decision Support")
    st.divider()

    st.subheader("⚙️ Decision Threshold")
    threshold = st.slider("Required confidence to flag risk", 0.3, 0.6, 0.35, 0.05)
    st.caption("Standard: 0.35. Increase for tighter intervention budgets.")

    st.divider()
    st.subheader("📊 Prediction Filter")
    pred_choice = st.selectbox("Type of employee to analyze", ["All", "Likely to Leave", "Likely to Stay"])
    want_leave_map = {"All": None, "Likely to Leave": True, "Likely to Stay": False}
    want_leave = want_leave_map[pred_choice]

    st.subheader("⭐ Employee Value")
    value_choice = st.selectbox("Priority tier", ["All", "High Value Employee", "Valuable Employee", "Average Employee", "Low Value Employee"])
    value_type = None if value_choice == "All" else value_choice

    st.divider()
    st.subheader("🎯 Risk Range")
    risk_range = st.slider("Probability filter (%)", 0, 100, (0, 100))
    min_risk, max_risk = risk_range[0] / 100, risk_range[1] / 100
    st.caption("Filter specific probability bands.")

# --- Header Section ---
st.title("🛡️ Employee Attrition Decision Support")
st.markdown("##### Strategic Intelligence Unit | Employee Retention Management")

# --- Control Center ---
if st.button("🔄 Generate Employee Scenario", type="primary", use_container_width=True):
    with st.spinner("Analyzing workforce data..."):
        analysis = run_strategic_analysis(model, df_raw, threshold, want_leave, value_type, min_risk, max_risk)
        if analysis:
            st.session_state.strategic_analysis = analysis
        else:
            st.warning("No employees matching your current filters were found in the database.")

# --- Dashboard Structure ---
tab1, tab2 = st.tabs(["📊 Strategic Dashboard", "🧠 Classification Logic"])

with tab1:
    if 'strategic_analysis' in st.session_state:
        res = st.session_state.strategic_analysis
        raw = res['raw']
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        # 1. Persona Profile (Left Card)
        ins = res['insights']
        
        # Ultra-compact modern High-fidelity layout
        def metric_card(label, value, icon=""):
            return f"""
            <div style="background-color: white; border: 1px solid #f1f5f9; border-radius: 8px; padding: 6px 10px; margin-bottom: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); height: 100%; border-left: 2px solid #3b82f611;">
                <div style="font-size: 9px; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 0px; letter-spacing: 0.01em;">{label}</div>
                <div style="font-size: 13px; color: #1e3a8a; font-weight: 700; display: flex; align-items: center; gap: 4px;">
                    <span style="font-size: 14px;">{icon}</span> {value}
                </div>
            </div>
            """

        with col1:
            with st.container(border=True):
                # Value-based styling
                val_meta = {
                    "High Value Employee": {"color": "#fbbf24", "icon": "⭐", "bg": "#fef3c7"},
                    "Valuable Employee": {"color": "#60a5fa", "icon": "💎", "bg": "#dbeafe"},
                    "Average Employee": {"color": "#94a3b8", "icon": "👤", "bg": "#f1f5f9"},
                    "Low Value Employee": {"color": "#f87171", "icon": "⚠️", "bg": "#fee2e2"}
                }.get(res["value_label"], {"color": "#94a3b8", "icon": "👤", "bg": "#f1f5f9"})

                st.markdown(f"### {raw['JobRole']}")
                
                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                        <span style="color: #64748b; font-weight: 500;">{raw['Department']} Department</span>
                        <span style="background-color: {val_meta['bg']}; color: {val_meta['color']}; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 700; border: 1px solid {val_meta['color']}44;">
                            {val_meta['icon']} {res['value_label'].upper()}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="section-header">📋 Demographics & Personal Details</div>', unsafe_allow_html=True)
                d_row1_col1, d_row1_col2, d_row1_col3 = st.columns(3)
                d_row1_col1.markdown(metric_card("Employee ID", f"#{raw.get('EmployeeNumber', 'N/A')}", "🆔"), unsafe_allow_html=True)
                d_row1_col2.markdown(metric_card("Age / Sex", f"{raw['Age']} ({raw.get('Gender', '?')[0]})", "👤"), unsafe_allow_html=True)
                d_row1_col3.markdown(metric_card("Marital Status", raw['MaritalStatus'], "💍"), unsafe_allow_html=True)
                
                d_row2_col1, d_row2_col2, d_row2_col3 = st.columns(3)
                d_row2_col1.markdown(metric_card("Commute Dist.", f"{raw['DistanceFromHome']} mi", "🏠"), unsafe_allow_html=True)
                d_row2_col2.markdown(metric_card("Business Travel", raw['BusinessTravel'].replace('Travel_', '').replace('_', ' '), "✈️"), unsafe_allow_html=True)
                d_row2_col3.markdown(metric_card("Stock Options", raw.get('StockOptionLevel', 0), "📊"), unsafe_allow_html=True)
                
                st.markdown('<div class="section-header">🏢 Career & Progress History</div>', unsafe_allow_html=True)
                c_row1_col1, c_row1_col2, c_row1_col3 = st.columns(3)
                c_row1_col1.markdown(metric_card("Monthly Salary", f"${raw['MonthlyIncome']:,}", "💰"), unsafe_allow_html=True)
                c_row1_col2.markdown(metric_card("Career Total", f"{raw.get('TotalWorkingYears', 0)}y", "📈"), unsafe_allow_html=True)
                c_row1_col3.markdown(metric_card("Latest Hike", f"{raw.get('PercentSalaryHike', 0)}%", "💹"), unsafe_allow_html=True)
                
                c_row2_col1, c_row2_col2, c_row2_col3 = st.columns(3)
                c_row2_col1.markdown(metric_card("Job Level", raw['JobLevel'], "🏗️"), unsafe_allow_html=True)
                c_row2_col2.markdown(metric_card("Last Promotion", f"{raw.get('YearsSinceLastPromotion', 0)}y ago", "⌛"), unsafe_allow_html=True)
                c_row2_col3.markdown(metric_card("Rating", f"{raw.get('PerformanceRating', 0)}/4", "🏆"), unsafe_allow_html=True)
                
                c_row3_col1, c_row3_col2, c_row3_col3 = st.columns(3)
                c_row3_col1.markdown(metric_card("Co. Tenure", f"{raw['YearsAtCompany']}y", "🏢"), unsafe_allow_html=True)
                c_row3_col2.markdown(metric_card("Overtime", '🚨 YES' if raw['OverTime'] == 'Yes' else 'No', "⏰"), unsafe_allow_html=True)
                c_row3_col3.markdown(metric_card("External Hist.", f"{raw['NumCompaniesWorked']} Cos.", "🏢"), unsafe_allow_html=True)

                st.markdown('<div class="section-header">🧠 Sentiment & Satisfaction Audit</div>', unsafe_allow_html=True)
                
                def sat_meta(v):
                    meta = {
                        1: ("🔴", "Low"),
                        2: ("🟠", "Med."),
                        3: ("🟡", "High"),
                        4: ("🟢", "V. High")
                    }.get(v, ("⚪", "N/A"))
                    return meta
                
                j_ico, j_lbl = sat_meta(raw.get('JobSatisfaction', 0))
                e_ico, e_lbl = sat_meta(raw.get('EnvironmentSatisfaction', 0))
                r_ico, r_lbl = sat_meta(raw.get('RelationshipSatisfaction', 0))
                m_tenure = raw.get('YearsWithCurrManager', 0)
                m_ico = "🟢" if m_tenure >= 3 else "🟡" if m_tenure >= 1 else "🔴"
                
                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                s_col1.markdown(metric_card("Job", j_lbl, j_ico), unsafe_allow_html=True)
                s_col2.markdown(metric_card("Env.", e_lbl, e_ico), unsafe_allow_html=True)
                s_col3.markdown(metric_card("Team", r_lbl, r_ico), unsafe_allow_html=True)
                s_col4.markdown(metric_card("Mgr.", f"{m_tenure}y", m_ico), unsafe_allow_html=True)
                
                st.markdown("""
                    <div style="background-color: #f8fafc; padding: 10px; border-radius: 8px; border: 1px dashed #cbd5e1; margin-top: 5px;">
                        <p style="font-size: 11px; color: #94a3b8; margin: 0;"><b>Job Sat.:</b> Specific role contentment | <b>Env. Qual.:</b> Workspace culture and safety</p>
                        <p style="font-size: 11px; color: #94a3b8; margin: 0;"><b>Team Rel.:</b> Peer collaboration quality | <b>Mgr. Tenure:</b> Stability under direct leadership</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="section-header">💡 Strategic Perspective Insights</div>', unsafe_allow_html=True)
                m_col1, m_col2, m_col3 = st.columns(3)
                
                m_col1.markdown(metric_card("Intensity", ins["Work Intensity"], "🔥"), unsafe_allow_html=True)
                m_col2.markdown(metric_card("Income Status", ins["Income Alignment"], "⚖️"), unsafe_allow_html=True)
                m_col3.markdown(metric_card("Stability", ins["Career Stability"], "🛡️"), unsafe_allow_html=True)

                st.markdown("""
                    <div style="background-color: #f8fafc; padding: 10px; border-radius: 8px; border: 1px dashed #cbd5e1; margin-top: 5px;">
                        <p style="font-size: 11px; color: #94a3b8; margin: 0;"><b>Intensity:</b> Workload stress vs life balance | <b>Income Status:</b> Salary vs role expectations</p>
                        <p style="font-size: 11px; color: #94a3b8; margin: 0;"><b>Stability:</b> Overall career tenure pattern and retention likelihood</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # 2. Risk Dynamics (Right)
        with col2:
            with st.container(border=True):
                
                # Risk Status Banner
                is_risky = res['prediction'] == "Leave"
                if is_risky:
                    st.error("🔴 **HIGH RISK — LIKELY TO LEAVE**", icon="🚨")
                else:
                    st.success("🟢 **LOW RISK — LIKELY TO STAY**", icon="✅")
                
                # Risk Distribution
                risk_color = "#dc2626" if is_risky else "#059669"
                st.markdown(f'<div class="risk-score" style="color:{risk_color};">{res["risk"]:.1%}</div>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align:center; color:#64748b; font-size:14px; margin-top:-10px;">Probability of Attrition</p>', unsafe_allow_html=True)
                st.progress(res['risk'])
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Precision Audit
                audit_col1, audit_col2, audit_col3 = st.columns(3)
                audit_col1.markdown(metric_card("Forecast", res['prediction'], "🔮"), unsafe_allow_html=True)
                audit_col2.markdown(metric_card("Reality", res['actual'], "👁️"), unsafe_allow_html=True)
                
                is_correct = res['prediction'] == res['actual']
                match_label = "Correct" if is_correct else "Incorrect"
                match_icon = "✅" if is_correct else "❌"
                audit_col3.markdown(metric_card("Audit Result", match_label, match_icon), unsafe_allow_html=True)
            
                st.divider()

                # Attrition Drivers (Grouped)
                st.markdown('<div class="section-header">🔍 Key Risk Drivers</div>', unsafe_allow_html=True)
                
                s_drivers = res['structured_drivers']
                main_pts = s_drivers.get("Main Drivers", ["No strong drivers"])
                detailed = s_drivers.get("Detailed Drivers", [])
                
                # Conditional display based on Prediction status
                if res['prediction'] == "Stay":
                    st.info("✅ **This employee is currently likely to stay.**")
                    if main_pts and main_pts[0] != "No strong drivers":
                        with st.expander("🔍 Potential Future Risks (Low Emphasis)"):
                             st.caption("These are secondary signals that do not currently indicate a threat but may warrant long-term monitoring.")
                             for pt in main_pts:
                                 st.markdown(f"**{pt.upper()}**")
                                 cat_upper = pt.upper()
                                 matches = [d for d in detailed if f"[{cat_upper}]" in d.upper()]
                                 for m in matches:
                                     _, factor = m.split("] ", 1) if "] " in m else ("", m)
                                     st.write(f"• {factor}")
                
                elif main_pts and main_pts[0] != "No strong drivers":
                    for pt in main_pts:
                        # Show Category Title (Red Badge)
                        st.markdown(f"""
                            <div style="background-color: #fce7e7; border: 1px solid #f87171; padding: 10px; border-radius: 8px 8px 0 0; margin-top: 10px; color: #991b1b; font-weight:700; text-align:center; font-size: 13px;">
                                🚨 {pt.upper()}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show Matching Detailed Factors (Blue Information Box)
                        cat_upper = pt.upper()
                        matches = [d for d in detailed if f"[{cat_upper}]" in d.upper()]
                        
                        if matches:
                            match_html = ""
                            for m in matches:
                                _, factor = m.split("] ", 1) if "] " in m else ("", m)
                                match_html += f'<div style="font-size: 12px; color: #1e40af; margin-bottom: 4px;">• {factor}</div>'
                            
                            st.markdown(f"""
                                <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-top: none; padding: 10px; border-radius: 0 0 8px 8px; margin-bottom: 15px;">
                                    {match_html}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                else:
                    st.success("No critical risk drivers identified based on current policy.")
                
                # Vertical balancing
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        st.divider()

        # 3. Decision Narrative (Strategic Summary)
        st.subheader("💡 Strategic Perspective")
        has_drivers = len(main_pts) > 0 and main_pts[0] != "No strong drivers"
        risk_summary = " and ".join([p.lower() for p in main_pts[:3]]) if has_drivers else "baseline factors"
        
        # Qualitative Risk intensity
        r = res['risk']
        if r < 0.10: risk_int = "NEGLIGIBLE"
        elif r < 0.25: risk_int = "LOW"
        elif r < 0.40: risk_int = "EMERGENT"
        elif r < 0.60: risk_int = "SIGNIFICANT"
        elif r < 0.85: risk_int = "HIGH RISK"
        else: risk_int = "SEVERE"
        
        # Strategic Guidance/Advice
        if r > threshold:
            # Identify top dynamic mitigations
            top_actions = [a.replace("⚠️ ", "").replace("✔️ ", "").replace("PRIORITY: ", "") for a in res['actions'][:2]]
            action_summary = ", ".join(top_actions).lower() if top_actions else "engagement and feedback improvements"

            if "High Value" in res['value_label']:
                advice = f"<b>Mission-Critical Retention.</b> Irreplaceable expertise at risk. <b>Mitigate via:</b> {action_summary}, and personalized leadership support."
            elif "Valuable" in res['value_label']:
                advice = f"<b>High Retention Priority.</b> Core stability asset at risk. <b>Mitigate via:</b> {action_summary}."
            elif "Average" in res['value_label']:
                advice = f"<b>General Mitigation.</b> Preventable risk detected. <b>Mitigate via:</b> {action_summary}."
            else:
                advice = f"<b>Strategic Evaluation.</b> Low-impact profile at high risk. <b>Recommendation:</b> Assess if the cost of retention/intervention exceeds the impact of departure. Focus on standard exit documentation."
        else:
            if r < 0.10:
                advice = "<b>Stable Engagement.</b> No attrition markers detected. Continue standard professional development rituals."
            else:
                advice = "<b>Stable Retention Outlook.</b> Minimal risk profile. Routine management interactions are sufficient to maintain current state."

        # Hardcoded styling for the Executive Summary (Always Red)
        summary_bg = "#fef2f2"
        summary_border = "#dc2626"
        
        # 3. Decision Narrative (Strategic Summary)
        # Narrative scaling based on risk
        risk_narrative = "driven primarily by" if r > threshold else "with minor signals observed in"
        advice_html = f"<li><b>Strategic Guidance:</b> {advice}</li>" if advice else ""
        
        st.markdown(f"""
        <div style="background-color: {summary_bg}; border-left: 5px solid {summary_border}; padding: 20px; border-radius: 8px;">
            <strong style="font-size: 18px;">Executive Summary:</strong><br>
            <div style="margin-top: 15px; font-size: 15px; color: #1e293b; line-height: 1.8;">
                <ul style="padding-left: 20px; margin: 0;">
                    <li><b>Workforce Classification:</b> Identified as a <b>{res['value_label']}</b> (based on tenure, performance, and role level).</li>
                    <li><b>Risk Assessment:</b> Detected a <b>{risk_int}</b> profile (<b>{r:.1%}</b> probability), {risk_narrative} <b>{risk_summary}</b>.</li>{advice_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # 4. Challenges vs. Actions (Side-by-Side)
        col_problems, col_actions = st.columns(2)
        
        with col_problems:
            st.subheader("❗ Identified Challenges")
            for p in res['problems']:
                st.markdown(f"- {p}")

        with col_actions:
            st.subheader("🎯 Retention Plan")
            for act in res['actions']:
                if "PRIORITY" in act:
                    st.markdown(f"<div style='background-color: #fffbeb; color: #b45309; padding: 6px 12px; border-radius: 6px; border: 1px solid #fde68a; font-weight: 700; font-size: 12px; display: inline-block; margin-bottom: 12px; border-left: 4px solid #f59e0b;'>⚠️ {act}</div>", unsafe_allow_html=True)
                else:
                    st.success(f"✔️ {act}")

                    
    else:
        # Empty State
        st.markdown("""
        <div style="text-align: center; padding: 100px; color: #64748b;">
            <h2 style="font-weight: 300;">Ready to analyze your workforce?</h2>
            <p>Click the button above to generate a strategic employee analysis scenario.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # --- Dynamic Logic for Current Employee ---
    if 'strategic_analysis' in st.session_state:
        res = st.session_state.strategic_analysis
        row = res['X_row']
        
        # Re-calculate points to identify specific path
        score_details = []
        if row['JobLevel'] >= 3: score_details.append("JL")
        if row['YearsAtCompany'] >= 5: score_details.append("TC")
        if row['Income_per_Level'] > 1: score_details.append("IE")
        if row['YearsWithCurrManager'] >= 3: score_details.append("MT")
        if row['PerformanceRating'] >= 4: score_details.append("PR")
        if row['JobInvolvement'] >= 3: score_details.append("JI")
        if row['YearsAtCompany'] < 2: score_details.append("ST")
        if row['NumCompaniesWorked'] >= 5: score_details.append("JH")
        if row['JobSatisfaction'] <= 2: score_details.append("LS")
        
        target_tier = {
            "High Value Employee": "HV",
            "Valuable Employee": "V",
            "Average Employee": "A",
            "Low Value Employee": "LV"
        }.get(res["value_label"], "A")

        # Generate Mermaid with Highlights
        hl_style = "stroke:#ef4444,stroke-width:4px"
        
        def get_style(key):
            return hl_style if key in score_details else ""

        st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 2rem;">
            <h2 style="color: #1e293b; margin-top: 0;">🧠 Personalized Classification Path</h2>
            <p style="color: #64748b; font-size: 1.1rem;">Below is the specific logical sequence used to classify <b>{res['raw']['JobRole']}</b> (Index {res['index']}) into the <b>{res['value_label']}</b> tier.</p>
        </div>
        """, unsafe_allow_html=True)

        # Generate Mermaid with Highlights
        hl_style = "stroke:#ef4444,stroke-width:4px"
        
        # Build style lines only if applicable
        style_lines = []
        for key, node_id in [("JL", "JL"), ("TC", "TC"), ("MT", "MT"), ("PR", "PR"), ("JI", "JI"), ("IE", "IE"), ("ST", "ST"), ("JH", "JH"), ("LS", "LS")]:
            if key in score_details:
                style_lines.append(f"    style {node_id} {hl_style}")
        
        # Style ALL possible end tiers, but highlight chosen one specially
        style_lines.append(f"    style {target_tier} fill:#fdf2f8,stroke:#ec4899,stroke-width:5px")
        style_lines.append(f"    style Score fill:#eff6ff,stroke:#3b82f6,stroke-width:3px")

        def get_arrow(key):
            return "==>" if key in score_details else "-->"

        style_section = "\n".join(style_lines)
        
        # Pre-calculate arrows to avoid syntax issues in Mermaid
        hv_arrow = '== ">= 5" ==>' if target_tier == "HV" else '-- ">= 5" -->'
        v_arrow = '== ">= 2" ==>' if target_tier == "V" else '-- ">= 2" -->'
        a_arrow = '== ">= 0" ==>' if target_tier == "A" else '-- ">= 0" -->'
        lv_arrow = '== "< 0" ==>' if target_tier == "LV" else '-- "< 0" -->'

        def get_edge(key, label):
            is_active = key in score_details
            return f'== "{label}" ==>' if is_active else f'-- "{label}" -->'

        mermaid_code = f"""
graph TD
    Start([Start: Base Score = 0]) --> Seniority{{Seniority & Role}}
    
    Seniority {get_edge('JL', '+2 pts')} JL["Job Level >= 3"]
    Seniority {get_edge('TC', '+2 pts')} TC["Tenure >= 5y"]
    Seniority {get_edge('MT', '+1 pt')} MT["High Mgr tenure"]
    
    JL {"==>" if 'JL' in score_details else "-->"} Perf{{Performance}}
    TC {"==>" if 'TC' in score_details else "-->"} Perf
    MT {"==>" if 'MT' in score_details else "-->"} Perf
    Seniority --> Perf
    
    Perf {get_edge('PR', '+2 pts')} PR["Rating >= 4"]
    Perf {get_edge('JI', '+1 pt')} JI["High Involvement"]
    Perf {get_edge('IE', '+1 pt')} IE["Income Efficiency"]
    
    PR {"==>" if 'PR' in score_details else "-->"} Status{{Status Factors}}
    JI {"==>" if 'JI' in score_details else "-->"} Status
    IE {"==>" if 'IE' in score_details else "-->"} Status
    Perf --> Status
    
    Status {get_edge('ST', '-2 pts')} ST["Short Tenure (< 2y)"]
    Status {get_edge('JH', '-1 pt')} JH["Job Hopping (5+)"]
    Status {get_edge('LS', '-1 pt')} LS["Low Job Sat (<= 2)"]
    
    ST {"==>" if 'ST' in score_details else "-->"} Score
    JH {"==>" if 'JH' in score_details else "-->"} Score
    LS {"==>" if 'LS' in score_details else "-->"} Score
    Status --> Score
    
    Score["Final Score = {res['value_score']:.1f}"]
    
    Score {hv_arrow} HV(High Value Employee)
    Score {v_arrow} V(Valuable Employee)
    Score {a_arrow} A(Average Employee)
    Score {lv_arrow} LV(Low Value Employee)
    
{style_section}
"""
        import streamlit.components.v1 as components
        html_code = f"""
        <div class="mermaid" style="display: flex; justify-content: center;">
        {mermaid_code.strip()}
        </div>
        <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true,
            securityLevel: 'loose',
            theme: 'base',
            themeVariables: {{
                'fontSize': '16px',
                'primaryColor': '#eff6ff',
                'primaryTextColor': '#1e3a8a',
                'primaryBorderColor': '#3b82f6',
                'lineColor': '#64748b',
                'secondaryColor': '#f8fafc',
                'tertiaryColor': '#ffffff'
            }},
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
        </script>
        """
        components.html(html_code, height=850, width=1200, scrolling=True)
        
        st.divider()
        
    else:
        # Generic Flow when no analysis generated
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 2rem;">
            <h2 style="color: #1e293b; margin-top: 0;">🧠 How We Classify Employee Value</h2>
            <p style="color: #64748b; font-size: 1.1rem;">Generate an employee scenario in the first tab to see their personalized classification path here.</p>
        </div>
        """, unsafe_allow_html=True)

        generic_mermaid = """
graph TD
    Start([Start: Base Score = 0]) --> Seniority{Seniority & Role}
    Seniority -->|+2: Job Level >= 3| Perf{Performance}
    Seniority -->|+2: Tenure >= 5y| Perf
    Seniority -->|+1: High Manager Tenure| Perf
    
    Perf -->|+2: Rating >= 4| Status{Status Factors}
    Perf -->|+1: High Involvement| Status
    Perf -->|+1: Income Efficiency| Status
    
    Status -->|-2: Tenure < 2y| Risk{Risk Factors}
    Status -->|-1: Job Hopping| Risk
    Status -->|-1: Low Satisfaction| Risk
    
    Risk --> Score[Final Score Calculation]
    
    Score -->|>= 5| HV[High Value Employee]
    Score -->|>= 2| V[Valuable Employee]
    Score -->|>= 0| A[Average Employee]
    Score -->|< 0| LV[Low Value Employee]
    
    style HV fill:#fef3c7,stroke:#fbbf24,stroke-width:2px
    style V fill:#dbeafe,stroke:#60a5fa,stroke-width:2px
    style A fill:#f1f5f9,stroke:#94a3b8,stroke-width:2px
    style LV fill:#fee2e2,stroke:#f87171,stroke-width:2px
"""
        import streamlit.components.v1 as components
        html_code_gen = f"""
        <div class="mermaid">
        {generic_mermaid.strip()}
        </div>
        <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
        </script>
        """
        components.html(html_code_gen, height=600, scrolling=True)

    col_rules1, col_rules2 = st.columns(2)

    with col_rules1:
        st.markdown("""
        <div style="background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 8px; height: 100%;">
            <h3 style="color: #065f46; margin-top: 0;">➕ Value Boosters (Positive Points)</h3>
            <ul style="color: #065f46; line-height: 1.6;">
                <li><b>Job Level (>= 3):</b> +2 pts (Senior Leadership Impact)</li>
                <li><b>Tenure (>= 5 years):</b> +2 pts (Deep Institutional Knowledge)</li>
                <li><b>High Performance (Rating >= 4):</b> +2 pts (Exceptional Output)</li>
                <li><b>Income Efficiency (> 1.0):</b> +1 pt (High Salary/Level Ratio)</li>
                <li><b>Manager Relationship (>= 3y):</b> +1 pt (Team Stability)</li>
                <li><b>Job Involvement (>= 3):</b> +1 pt (Strong Active Engagement)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_rules2:
        st.markdown("""
        <div style="background-color: #fff1f2; border-left: 5px solid #f43f5e; padding: 1.5rem; border-radius: 8px; height: 100%;">
            <h3 style="color: #9f1239; margin-top: 0;">⚠️ Risk & Adjustment Factors (Negative Points)</h3>
            <ul style="color: #9f1239; line-height: 1.6;">
                <li><b>Low Tenure (< 2 years):</b> -2 pts (Higher Initial Attrition Risk)</li>
                <li><b>Job Hopping (>= 5 cos):</b> -1 pt (Historical Stability Risk)</li>
                <li><b>Low Satisfaction (<= 2):</b> -1 pt (Immediate Disengagement Risk)</li>
            </ul>
            <p style="color: #9f1239; font-style: italic; margin-top: 1rem;">These factors weigh against the positive scores to provide a balanced view of employee "stickiness" and value.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f1f5f9; padding: 2rem; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h3 style="text-align: center; color: #334155; margin-bottom: 2rem;">🏆 Final Value Tiers</h3>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
            <div style="background-color: #fef3c7; border: 2px solid #fbbf24; padding: 1rem; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 2rem;">⭐</div>
                <h4 style="color: #92400e; margin: 0.5rem 0;">High Value</h4>
                <p style="font-weight: 700; color: #b45309; font-size: 1.2rem;">Score >= 5</p>
                <p style="font-size: 0.8rem; color: #d97706;">Mission-critical talent, key leadership.</p>
            </div>
            <div style="background-color: #dbeafe; border: 2px solid #60a5fa; padding: 1rem; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 2rem;">💎</div>
                <h4 style="color: #1e40af; margin: 0.5rem 0;">Valuable</h4>
                <p style="font-weight: 700; color: #1d4ed8; font-size: 1.2rem;">Score >= 2</p>
                <p style="font-size: 0.8rem; color: #2563eb;">Stable core talent, high potential.</p>
            </div>
            <div style="background-color: #f1f5f9; border: 2px solid #94a3b8; padding: 1rem; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 2rem;">👤</div>
                <h4 style="color: #475569; margin: 0.5rem 0;">Average</h4>
                <p style="font-weight: 700; color: #64748b; font-size: 1.2rem;">Score >= 0</p>
                <p style="font-size: 0.8rem; color: #475569;">Solid contributors, standard performance.</p>
            </div>
            <div style="background-color: #fee2e2; border: 2px solid #f87171; padding: 1rem; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 2rem;">⚠️</div>
                <h4 style="color: #991b1b; margin: 0.5rem 0;">Low Value</h4>
                <p style="font-weight: 700; color: #b91c1c; font-size: 1.2rem;">Score < 0</p>
                <p style="font-size: 0.8rem; color: #dc2626;">At-risk performance or low engagement.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
