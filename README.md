# 🧠 HR Attrition Intelligence Hub
### **Bridging Data Science with Structured Business Logic**

---

## 📺 Live Demo: Attrition Decision Support System
**Experience the live strategic dashboard here: [https://employee-retention-ai.streamlit.app/](https://employee-retention-ai.streamlit.app/)**

![Strategic Dashboard Demo](assets/dashboard_demo.png)

---

## 🗺️ System Architecture: The Big Picture
We combine data-driven prediction with structured business logic to turn raw workforce insights into high-impact retention actions.

```mermaid
graph TD
    subgraph "Strategic Drivers"
        W[Workload Stress]
        L[Life Stage Mobility]
        S[Seniority & Compensation]
        R[Role Structure]
    end

    subgraph "Training Pipeline (Back-End)"
        D1[Raw HR Data] --> P1[Pre-processing]
        P1 --> F1[Feature Engineering]
        F1 --> M1[Model Selection]
        M1 --> T1[Final Model Training]
    end

    T1 -- "Deployed Model" --> P2

    subgraph "Inference Intelligence (Front-End)"
        D2[New Employee Data] --> P2[Model Prediction]
        P2 --> S2[Employee Value Scoring]
        S2 --> R2[Risk Driver Identification]
        R2 --> F2[Strategic Problem Framing]
    end

    F2 --> O([Retention Actions / Decisions])

    style O fill:#f0fdf4,stroke:#16a34a,stroke-width:3px
    style S2 fill:#eff6ff,stroke:#3b82f6,stroke-width:2px
    style R2 fill:#fff1f2,stroke:#f43f5e,stroke-width:2px
```

---

## 💎 1. Employee Value Scoring Logic
Not all attrition is equal. Our system prioritizes retaining high-impact talent by scoring employees based on performance, growth velocity, and seniority.

**Logic Transparency in Action:**
![Classification Logic View](assets/classification_logic.png)

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
    
    style HV fill:#fef3c7,stroke:#fbbf24,stroke-width:2px
    style LV fill:#fee2e2,stroke:#f87171,stroke-width:2px
```

---

## 🔍 2. Risk Identification & Problem Framing
We translate raw employee features into clear business problems using a multi-stage intelligence pipeline.

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
    style RA fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
```

---

## 🎯 3. From Risk Drivers to Retention Actions
The system maps identified strategic problems to specific, actionable HR interventions to ensure consistency and speed in retention efforts.

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
*   **Predictive Model:** Random Forest / Gradient Boosting (Trained on IBM HR Attrition).
*   **Logic Engine:** Hierarchical thresholding and weighted value classification.
*   **Interface:** Streamlit (Custom Executive UI) with real-time Mermaid.js visualizations.

---

Developed by **Andrew Hany**. 
*Turning Workforce Data into Strategic Talent Retention.*
