#  HR Attrition Intelligence Hub
### **Bridging Data Science with Structured Business Logic**
##  Problem & Value

Employee attrition is costly and difficult to manage proactively.

Most solutions stop at prediction — this system goes further by:
- Explaining *why* employees are at risk
- Prioritizing *who* to retain
- Recommending *what actions* to take

This project transforms attrition prediction into a **decision support system for HR strategy**.
---

##  Live Demo: Attrition Decision Support System
**Experience the live strategic dashboard here: [https://employee-retention-ai.streamlit.app/](https://employee-retention-ai.streamlit.app/)**

![Strategic Dashboard Demo](assets/dashboard_demo.png)

---


## ⚡ Quick Start

```bash
git clone https://github.com/Andrew-Hany/employee-attrition-decision-system.git
cd employee-attrition-decision-system
pip install -r requirements.txt
streamlit run app.py
```
---

---

## 📊 Phase 1: Exploratory Data Analysis (EDA)
Our initial discovery (detailed in [EDA.ipynb](EDA.ipynb)) revealed that attrition is driven by **four core strategic drivers**. By analyzing these sub-metrics, we can move from raw data to a strategic "Employee Value" classification.

### **The Attrition Driver Tree**
```mermaid
graph LR
    D[Attrition Drivers] --> W[Workload Stress]
    D --> C[Seniority & Comp.]
    D --> L[Life Stage Mobility]
    D --> R[Role Design]
    
    W --> W1[Overtime]
    W --> W2[Job Involvement]
    
    C --> C1[Job Level / Salary]
    C --> C2[Income Efficiency]
    C --> C3[Promotion Velocity]
    
    L --> L1[Age / Experience]
    L --> L2[Marital Status]
    L --> L3[Jump History]
    
    R --> R1[Travel Load]
    R --> R2[Job Satisfaction]
    R --> R3[Environment Fit]

    style D fill:#f8fafc,stroke:#334155,stroke-width:2px
    classDef pillar fill:#eff6ff,stroke:#3b82f6,stroke-width:1px;
    class W,C,L,R pillar
```

### **Strategic Pillar Details**
1.  **Workload:** Impact of overtime vs. role involvement.
2.  **Seniority & Compensation:** Evaluation of salary competitiveness (Income Efficiency) and growth rate (Promotion Velocity).
3.  **Life Stage:** Factors in external tenure stability and professional mobility.
4.  **Role Structure:** Focuses on travel burden and environmental satisfaction.

---


## ⚙️ Phase 2: Model Training Pipeline
The technical development of our predictive engine is detailed in [attrition_model.ipynb](attrition_model.ipynb). This stage focuses on transforming raw behavior features into a high-recall retention model.

### **The Training Workflow**
```mermaid
graph TD
    subgraph "Training Lifecycle"
        D1[Raw IBM HR Data] --> P1[Data Pre-processing]
        P1 --> F1[Feature Engineering]
        F1 --> M1[Model Selection & Tuning]
        M1 --> T1[Final Model Training]
    end
    
    style T1 fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
```

### **Model Selection & Performance Results**
We prioritized **Recall** (catching as many potential leavers as possible) while maintaining a balanced **Precision**. As shown below, **SMOTE (Oversampling)** was the key breakthrough in our modeling strategy.

| Experiment | Model | Threshold | Test ROC AUC | Recall (Leave) | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I. Baseline** | **Logistic Regression** | 0.35 | **0.810** | 0.489 | **0.535** | **0.511** |
| (No SMOTE) | CatBoost | 0.35 | 0.780 | 0.511 | 0.453 | 0.480 |
| | Random Forest | 0.40 | 0.779 | 0.170 | 0.444 | 0.246 |
| <hr> | <hr> | <hr> | <hr> | <hr> | <hr> | <hr> |
| **II. SMOTE** | ✨ **Logistic Regression** | **0.40** | **0.789** | 🚀 **0.766** | 0.353 | **0.483** |
| (Over-sampled) | Random Forest | 0.35 | 0.771 | 0.723 | 0.333 | 0.456 |
| | CatBoost | 0.35 | 0.758 | 0.681 | 0.327 | 0.441 |

> [!TIP]
> **Key Insight:** Moving from Baseline to SMOTE increased our ability to catch leavers (Recall) from **48.9%** to **76.6%**, which is critical for a proactive retention strategy.

### **Threshold Optimization**
The system uses a **Logistic Regression** model optimized at a **0.35 - 0.40 threshold**.
*   **Lower Threshold (0.35):** Maximizes Recall (Catching more potential leavers).
*   **Selection:** We chose **0.40** for the baseline to achieve a **76.6% Recall rate**.

---

## 🎯 Phase 3: Inference Strategy (Intelligence Hub)
The final system takes the trained model and wraps it in a **Human-Centric Intelligence Layer**. This hub transforms raw numerical predictions into a structured strategic narrative.

### **The Inference Pipeline**
```mermaid
graph TD
    subgraph "Live Strategy Engine"
        D2[New Employee Data] --> P2[Model Prediction]
        P2 --> S2[Employee Value Scoring]
        S2 --> R2[Risk Driver Identification]
        R2 --> F2[Problem Framing]
        F2 --> O([Retention Actions / Decisions])
    end
    
    style O fill:#f0fdf4,stroke:#16a34a,stroke-width:3px
```

---

### **1. Strategic Value Scoring Logic**
The system prioritizes retention for high-impact talent by processing model outputs through our **Value Scoring Engine**.

```mermaid
graph TD
    Start([Start: Base Score = 0]) --> Seniority[Seniority & Role]
    
    Seniority --> JL["+2: Job Level >= 3"]
    Seniority --> TC["+2: Tenure >= 5y"]
    Seniority --> MT["+1: High Mgr Tenure"]
    
    JL --> Perf{Performance}
    TC --> Perf
    MT --> Perf
    
    Perf --> PR3["+1: Rating == 3"]
    Perf --> PR4["+2: Rating >= 4"]
    Perf --> JI["+1: Involvement >= 3"]
    Perf --> IE["+1: Income per Level > 3000"]
    Perf --> PV["+1: Promotion Velocity > 2.5"]
    
    PR3 & PR4 & JI & IE & PV --> Status{Status Factors}
    
    Status --> ST["-2: Tenure < 2y"]
    Status --> JH["-1: Job Hopping (5+)"]
    Status --> LS["-1: Low Satisfaction (<= 2)"]
    
    ST & JH & LS --> Score[Final Score Calculation]
    
    Score --> HV([High Value Employee])
    Score --> V([Valuable Employee])
    Score --> A([Average Employee])
    Score --> LV([Low Value Employee])

    %% Design Styles
    style Start fill:#f8fafc,stroke:#64748b,stroke-width:1px
    style Seniority fill:#f8fafc,stroke:#64748b,stroke-width:1px
    style Perf fill:#eff6ff,stroke:#3b82f6,stroke-width:2px
    style Status fill:#eff6ff,stroke:#3b82f6,stroke-width:2px
    style Score fill:#f8fafc,stroke:#64748b,stroke-width:1px
    
    classDef booster fill:#eff6ff,stroke:#3b82f6,stroke-width:1px,color:#1e40af;
    class JL,TC,MT,PR3,PR4,JI,IE,PV booster
    
    classDef risk fill:#fff1f2,stroke:#f43f5e,stroke-width:1px,color:#9f1239;
    class ST,JH,LS risk

    style HV fill:#fef3c7,stroke:#fbbf24,stroke-width:3px
    style V fill:#dbeafe,stroke:#60a5fa,stroke-width:3px
    style A fill:#f1f5f9,stroke:#94a3b8,stroke-width:3px
    style LV fill:#fee2e2,stroke:#f87171,stroke-width:3px
```

---

### **2. Risk Identification & Problem Framing**
This layer translates complex feature correlations into clear, human-readable **Business Problem Frames**.

```mermaid
graph TD
    A[Employee Features] --> B[Rule-Based Conditions]
    B --> C[Triggered Signals]
    C --> D{Driver Categories}
    
    D --> W[Workload Stress]
    D --> Co[Seniority & Comp.]
    D --> Ro[Role Structure]
    D --> Li[Life Stage]
    
    W & Co & Ro & Li --> SD[Structured Drivers]
    SD --> PF[Problem Framing]
    PF --> RA([Retention Actions / Decisions])
    
    style PF fill:#eff6ff,stroke:#3b82f6,stroke-width:2px
    style RA fill:#f0fdf4,stroke:#16a34a,stroke-width:3px
```

---

### **3. Strategic Action Mapping**
Identified problems are linked directly to targeted HR interventions to ensure immediate, data-backed response.

```mermaid
graph TD
    PF[Problem Framing] --> BR[Burnout Risk]
    PF --> CM[Career Misalignment]
    PF --> TR[Travel Burden]
    PF --> PR[Personal Mobility Risk]
    
    BR --> A1[Reduce Workload]
    BR --> A2[Improve Work-Life Balance]
    CM --> A3[Review Compensation]
    CM --> A4[Define Promotion Path]
    TR --> A5[Adjust Role Design]
    TR --> A6[Reduce Travel]
    PR --> A7[Remote Work]
    PR --> A8[Relocation Support]

    classDef action fill:#f0fdf4,stroke:#16a34a,stroke-width:1px;
    class A1,A2,A3,A4,A5,A6,A7,A8 action
    classDef prob fill:#fff1f2,stroke:#f43f5e,stroke-width:1px;
    class BR,CM,TR,PR prob
```

---

## 🛠️ Technical Stack

- **Model:** Logistic Regression (optimized with SMOTE)
- **Explainability Layer:** Rule-based driver identification
- **Decision Engine:** Value scoring + problem framing
- **Interface:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Mermaid

## 📂 Project Structure

- EDA.ipynb → Exploratory analysis
- attrition_model.ipynb → Model training & evaluation
- app.py → Streamlit application
- utils/ → Logic (scoring, drivers, recommendations)
- assets/ → Images & visuals
---

Developed by **Andrew Zaki**. 

