# 🧠 HR Attrition Intelligence Hub

### **Strategic Employee Retention & Decision Support System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://employee-retention-ai.streamlit.app)

---

## 📌 Executive Summary
Most HR tools tell you *who* might leave. This system tells you **why** they are leaving and **exactly what to do about it.**

By bridging Machine Learning (IBM HR Analytics) with a custom **Strategic Logic Layer**, this platform classifies workforce value, detects toxic risk combinations (like Burnout + Disengagement), and generates automated retention plans tailored to each employee’s unique profile.

---

## 🚀 Key Features

### 1. **Strategic Employee Value Classification**
Beyond basic performance, the system uses a **weighted point-based algorithm** to categorize employees into four tiers:
*   ⭐ **High Value:** Mission-critical talent with high seniority and growth velocity.
*   💎 **Valuable:** Stable core contributors with high potential.
*   👤 **Average:** Solid performers meeting standard expectations.
*   ⚠️ **Low Value:** Disengaged or low-tier impact roles requiring strategic evaluation.

### 2. **Intelligent Risk Driver Engine**
Developed using complex feature engineering, the system identifies **root causes** across four high-signal categories:
*   **Workload:** Detects "Toxic Combos" (e.g., *High Overtime + Low Involvement*).
*   **Seniority & Compensation:** Analyzes **Income Efficiency** (Salary normalized by Job Level) and **Promotion Velocity**.
*   **Role Structure:** Monitors travel burden and role stability.
*   **Life Stage:** Factors in age and external market mobility risk.

### 3. **Personalized "Logic-on-Path" Visualization**
Total transparency for HR leads. Every prediction is accompanied by a **dynamic Mermaid.js flowchart** that highlights the exact logical sequence used to classify that specific employee.

### 4. **Actionable Strategic Guidance**
Direct bridge from prediction to intervention. The system generates:
*   **Problem Detection:** Clear "Human-Readable" labels (e.g., *Detected high intensity burnout paired with total disengagement*).
*   **Retention Actions:** Specific, tiered commands (e.g., *Immediate re-engagement talk*, *Redistribute urgent tasks*).

---

## 🛠️ Technical Stack

*   **Core Logic:** Python 3.x
*   **Machine Learning:** Scikit-Learn (Random Forest/XGBoost logic)
*   **Feature Engineering:** Custom normalization of salary-to-level ratios and growth velocity metrics.
*   **Interface:** Streamlit (Custom CSS for premium Dashboard UI)
*   **Visuals:** Mermaid.js (SVG logic trees)
*   **Data Source:** IBM HR Analytics Attrition Dataset

---

## 📦 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Andrew-Hany/employee-attrition-decision-system.git
cd employee-attrition-decision-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Dashboard:**
```bash
streamlit run app.py
```

---

## 📊 Methodology: The "Value" Algorithm
The system goes beyond raw 'Attrition %' by calculating a **Combined Value Score**:
*   **Seniority (+2):** Job Level >= 3
*   **Loyalty (+2):** Tenure >= 5 years
*   **Growth (+1):** Promotion Velocity > 2.5 (Tenure / Promotion ratio)
*   **Efficiency (+1):** Income per Level > 3000 (Normalized salary competitiveness)
*   **Risk Adjustments:** Penalties for low tenure (< 2y) or high job-hopping history.

---

## 🤝 Contact & Contribution
Developed by **Andrew Hany**. Feel free to reach out for collaboration or strategic HR analytics consultations!
