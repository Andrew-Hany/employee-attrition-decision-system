import pandas as pd
import random
import joblib

# --- Columns dropped during training ---
DROP_COLS = ["Over18", "EmployeeCount", "StandardHours"]


# --- Preprocessing ---
def preprocess_input(df):
    df = df.copy()
    
    # drop same cols as training
    df = df.drop(columns=DROP_COLS, errors='ignore')
    
    # encoding
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
        
    if 'OverTime' in df.columns:
        df['OverTime'] = df['OverTime'].map({'No':0,'Yes':1})
    
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    
    # feature engineering
    df['Income_per_Level'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    df['Years_per_Promotion'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
    df['Company_Stability'] = df['YearsAtCompany'] / (df['NumCompaniesWorked'] + 1)
    df['Work_Intensity'] = df['OverTime'] * (5 - df['WorkLifeBalance'])
    df['Travel_Burden'] = df.get('BusinessTravel_Travel_Frequently', 0) * df['DistanceFromHome']
    
    df = df.drop(columns=['MonthlyIncome'], errors='ignore')
    
    return df


# --- Load Model ---
def load_model(path="attrition_model.pkl"):
    return joblib.load(path)


# --- Prediction ---
def predict_attrition(model, X_row, threshold=0.35):
    proba = model.predict_proba(X_row)[0][1]
    pred = int(proba > threshold)
    return pred, proba


# --- Employee Value ---
def classify_employee_value(row):
    
    score = 0
    
    if row['JobLevel'] >= 3: score += 2
    if row['YearsAtCompany'] >= 5: score += 2
    if row['Income_per_Level'] > 1: score += 1
    if row['YearsWithCurrManager'] >= 3: score += 1
    
    if row['PerformanceRating'] >= 4: score += 2
    if row['JobInvolvement'] >= 3: score += 1
    
    if row['YearsAtCompany'] < 2: score -= 2
    if row['NumCompaniesWorked'] >= 5: score -= 1
    if row['JobSatisfaction'] <= 2: score -= 1

    if score >= 5:
        label = "High Value Employee"
    elif score >= 2:
        label = "Valuable Employee"
    elif score >= 0:
        label = "Average Employee"
    else:
        label = "Low Value Employee"

    return score, label


def sample_employee(
    model, 
    raw_df, 
    threshold=0.35, 
    want_leave=None, 
    value_type=None,
    min_risk=None,
    max_risk=None
):
    
    processed_df = preprocess_input(raw_df)
    processed_df = processed_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    probas = model.predict_proba(processed_df)[:, 1]
    preds = (probas > threshold).astype(int)
    
    candidates = []
    
    for i in range(len(raw_df)):
        pred = preds[i]
        proba = probas[i]
        
        row_processed = processed_df.iloc[i]
        _, label = classify_employee_value(row_processed)
        
        # --- risk range filter ---
        if min_risk is not None and proba < min_risk:
            continue
        if max_risk is not None and proba > max_risk:
            continue
        
        # --- leave filter ---
        if want_leave is not None:
            if want_leave and pred != 1: continue
            if not want_leave and pred != 0: continue
        
        # --- value filter ---
        if value_type is not None and label != value_type:
            continue
        
        candidates.append({
            "X_row": processed_df.iloc[[i]],
            "index": i,
            "pred": pred,
            "proba": proba
        })
    
    return random.choice(candidates) if candidates else None

def get_attrition_drivers(row):

    DRIVER_MAP = {
        "Workload": ["OverTime", "WorkLifeBalance", "JobInvolvement", "Work_Intensity"],
        "Seniority & Compensation": ["Income_per_Level", "YearsSinceLastPromotion", "Stagnation", "Company_Stability"],
        "Role Structure": ["FrequentTravel", "Travel_Stress"],
        "Life Stage": ["Age", "MaritalStatus_Single", "DistanceFromHome"]
    }

    CONDITIONS = {
        # --- Workload ---
        "OverTime": lambda r: r.get("OverTime", 0) == 1,
        "WorkLifeBalance": lambda r: r.get("WorkLifeBalance", 3) <= 2,
        "JobInvolvement": lambda r: r.get("JobInvolvement", 3) <= 2,
        "Work_Intensity": lambda r: r.get("Work_Intensity", 0) > 2,

        # --- Career ---
        "Income_per_Level": lambda r: r.get("Income_per_Level", 1) < 1,
        "YearsSinceLastPromotion": lambda r: r.get("YearsSinceLastPromotion", 0) >= 3,
        "Stagnation": lambda r: r.get("YearsAtCompany", 0) >= 4 and r.get("JobLevel", 1) <= 2 and r.get("YearsSinceLastPromotion", 0) >= 2,
        "Company_Stability": lambda r: r.get("Company_Stability", 2) < 1,

        # --- Role ---
        "FrequentTravel": lambda r: r.get("BusinessTravel_Travel_Frequently", 0) == 1,
        "Travel_Stress": lambda r: r.get("BusinessTravel_Travel_Frequently", 0) == 1 and r.get("DistanceFromHome", 0) > 15,

        # --- Life ---
        "Age": lambda r: r.get("Age", 35) < 25,
        "MaritalStatus_Single": lambda r: r.get("MaritalStatus_Single", 0) == 1,
        "DistanceFromHome": lambda r: r.get("DistanceFromHome", 10) > 15
    }

    EXPLANATIONS = {
        "OverTime": "Frequent overtime (burnout risk)",
        "WorkLifeBalance": "Poor work-life balance",
        "Work_Intensity": "High workload intensity",
        "JobInvolvement": "Low engagement (underutilization)",

        "Income_per_Level": "Compensation below role expectations",
        "YearsSinceLastPromotion": "No recent promotion",
        "Stagnation": "Long tenure without advancement",
        "Company_Stability": "Low career stability",

        "FrequentTravel": "Frequent business travel",
        "Travel_Stress": "Travel + long commute strain",

        "Age": "Early career stage",
        "MaritalStatus_Single": "Higher mobility",
        "DistanceFromHome": "Long commute"
    }

    driver_scores = {}
    detailed_drivers = []
    workload_high = 0
    workload_low = 0

    for driver, features in DRIVER_MAP.items():
        score = 0
        
        for feat in features:
            if feat in CONDITIONS and CONDITIONS[feat](row):
                score += 1
                detailed_drivers.append(f"[{driver.upper()}] {EXPLANATIONS.get(feat, feat)}")

                # 🔥 workload direction logic
                if driver == "Workload":
                    if feat == "JobInvolvement":
                        workload_low += 1
                    else:
                        workload_high += 1
        
        if score > 0:
            driver_scores[driver] = score

    sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
    main_drivers = [d[0] for d in sorted_drivers[:3]]

    # 🔥 workload interpretation
    workload_type = None
    if workload_high > workload_low:
        workload_type = "High workload (burnout risk)"
    elif workload_low > workload_high:
        workload_type = "Low engagement (underutilization)"

    return {
        "Main Drivers": main_drivers if main_drivers else ["No strong drivers"],
        "Detailed Drivers": list(set(detailed_drivers)),
        "Workload Type": workload_type
    }
def recommend_actions(row, risk, value_label, drivers, threshold=0.35):

    main = drivers["Main Drivers"]
    details = drivers["Detailed Drivers"]
    workload_type = drivers.get("Workload Type")

    if risk < threshold:
        return {
            "Problems": ["No critical issues detected"],
            "Actions": ["Maintain current conditions and monitor periodically"]
        }

    problems = []
    actions = []

    # --- Workload (smart split) ---
    if "Workload" in main:
        if workload_type == "High workload (burnout risk)":
            problems.append("Attrition driven by high workload and burnout risk.")
            actions.append("Reduce workload and improve work-life balance")
        elif workload_type == "Low engagement (underutilization)":
            problems.append("Detected low engagement indicating role underutilization.")
            actions.append("Increase responsibility and role engagement")

    # --- Compensation ---
    if "Seniority & Compensation" in main:
        problems.append("Evidence of compensation or career progression concerns.")
        actions.append("Review promotion path and compensation alignment")

    # --- Role ---
    if "Role Structure" in main:
        problems.append("Role demands or travel burden affecting retention.")
        actions.append("Adjust role design or reduce travel requirements")

    # --- Life ---
    if "Life Stage" in main:
        problems.append("Personal life stage factors impacting employee mobility.")
        actions.append("Offer flexibility (remote work, relocation support)")

    # --- Detail boosters ---
    if any("promotion" in d.lower() for d in details):
        actions.append("Create clear career development plan")

    if any("overtime" in d.lower() for d in details):
        actions.append("Limit overtime and redistribute workload")

    # --- Priority ---
    if value_label == "High Value Employee":
        actions.insert(0, "PRIORITY: Immediate retention action required")

    return {
        "Problems": problems if problems else ["No major issues identified"],
        "Actions": list(set(actions))
    }
# # --- Full System ---
def attrition_system(model, df, threshold=0.35, want_leave=None, value_type=None, min_risk=None, max_risk=None):
    
    sample = sample_employee(model, df, threshold, want_leave, value_type, min_risk, max_risk)
    
    if sample is None:
        return {"Error": "No matching employee found"}
    
    idx = sample["index"]
    X_row = sample["X_row"]
    pred = sample["pred"]
    proba = sample["proba"]
    
    true_raw = df.loc[idx, 'Attrition']
    actual = "Leave" if true_raw == "Yes" or true_raw == 1 else "Stay"
    
    score, value_label = classify_employee_value(X_row.iloc[0])
    
    drivers = get_attrition_drivers(X_row.iloc[0])
    recs = recommend_actions(X_row.iloc[0], proba, value_label, drivers, threshold)
    
    return {
        "AttritionRisk": round(proba, 3),
        "Prediction": "Leave" if pred == 1 else "Stay",
        "Actual": actual,
        "ValueScore": score,
        "EmployeeType": value_label,
        "Drivers": drivers,
        "Problems": recs["Problems"],
        "Actions": recs["Actions"]
    }


# # --- Run ---
# if __name__ == "__main__":
    
#     model = load_model()
#     df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
#     result = attrition_system(
#         model,
#         df,
#         threshold=0.35, 
#         want_leave=None, 
#         value_type=None, 
#         min_risk=None, 
#         max_risk=None
#     )
    
#     print(result)