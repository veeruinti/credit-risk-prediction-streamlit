import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ================= LOAD SAVED PIPELINE =================
@st.cache_resource
def load_pipeline():
    with open("credit_final_model.pkl", "rb") as f:
        deploy_dict = pickle.load(f)
    return deploy_dict

deploy_dict = load_pipeline()


# ================= PREPROCESS FUNCTION (same as your notebook) =================
def preprocess_unseen(raw_df: pd.DataFrame, deploy_dict: dict) -> np.ndarray:
    df = raw_df.copy()

    mode_income = deploy_dict["mode_MonthlyIncome"]
    # Remove duplicate income column if exists
    df.drop(columns=["MonthlyIncome.1"], errors="ignore", inplace=True)
    
    # Impute main MonthlyIncome → create MonthlyIncome_mode
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(mode_income)
    df["MonthlyIncome_mode"] = df["MonthlyIncome"]
    
    df.drop(columns=["MonthlyIncome"], inplace=True)


    # 2) NumberOfDependents handling
    med_dep = deploy_dict["median_NumberOfDependents"]
    df["NumberOfDependents"] = pd.to_numeric(df["NumberOfDependents"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(med_dep)
    df["NumberOfDependents_median"] = df["NumberOfDependents"]
    df.drop(columns=["NumberOfDependents"], errors="ignore", inplace=True)

    # 3) Numeric log features (no clipping)
    log_map = {
        "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilizationOfUnsecuredLines_log_5th",
        "age": "age_log_5th",
        "DebtRatio": "DebtRatio_log_5th",
        "NumberOfOpenCreditLinesAndLoans": "NumberOfOpenCreditLinesAndLoans_log_5th",
        "NumberRealEstateLoansOrLines": "NumberRealEstateLoansOrLines_log_5th",
        "MonthlyIncome_mode": "MonthlyIncome_mode_log_5th",
        "NumberOfDependents_median": "NumberOfDependents_median_log_5th",
    }

    for raw_col, new_col in log_map.items():
        df[new_col] = np.log(df[raw_col] + 1)

    # 4) Categorical encoding
    oh_gender  = deploy_dict["one_hot_gender"]
    oh_region  = deploy_dict["one_hot_region"]
    od_rented  = deploy_dict["ordinal_rented"]
    od_occ     = deploy_dict["ordinal_occupation"]
    od_edu     = deploy_dict["ordinal_education"]

    # Gender -> Gender_male
    gender_arr = oh_gender.transform(df[["Gender"]]).toarray()
    df["Gender_male"] = gender_arr[:, 1].astype(int)

    # Region -> Central / East / North / South
    region_arr = oh_region.transform(df[["Region"]]).toarray()
    region_cats = list(oh_region.categories_[0])
    for i, cat in enumerate(region_cats):
        df[cat] = region_arr[:, i].astype(int)

    # Rented_OwnHouse -> Rented
    df["Rented"] = od_rented.transform(df[["Rented_OwnHouse"]]).astype(int)

    # Occupation -> Occupation_re
    df["Occupation_re"] = od_occ.transform(df[["Occupation"]]).astype(int)

    # Education -> Education_re
    df["Education_re"] = od_edu.transform(df[["Education"]]).astype(int)

    # Drop original categorical columns
    df.drop(
        columns=["Gender", "Region", "Rented_OwnHouse", "Occupation", "Education"],
        inplace=True,
        errors="ignore"
    )

    # 5) Keep only the final training features
    feature_names = deploy_dict["feature_names"]
    X = df[feature_names].copy()

    # 6) Scale
    scaler = deploy_dict["scaler"]
    X_scaled = scaler.transform(X)

    return X_scaled


def predict_unseen(raw_df: pd.DataFrame, deploy_dict: dict) -> pd.DataFrame:
    X_scaled = preprocess_unseen(raw_df, deploy_dict)
    model = deploy_dict["model"]
    le    = deploy_dict["label_encoder"]

    y_pred = model.predict(X_scaled)
    y_labels = le.inverse_transform(y_pred)

    result = raw_df.copy()
    result["Predicted_Label"] = y_labels

    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(X_scaled)
        classes = list(le.classes_)        # e.g. ['Bad', 'Good']
        good_index = classes.index("Good")

        result["Prob_Good"] = proba_all[:, good_index]
    else:
        result["Prob_Good"] = np.nan

    return result


# ================= STREAMLIT UI =================
st.title("Credit fraud Prediction")
st.write("Enter customer details to check if the credit is **Good** or risky.")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        npa_status = st.number_input("NPA Status", value=0.0)
        ruul = st.number_input("Revolving Utilization Of Unsecured Lines", min_value=0.0, value=0.25)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Central", "East", "North", "South"])
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
        education = st.text_input("Education", value="Graduate")
        rented_ownhouse = st.selectbox("Rented / Own House", ["Rented", "Own"])

    with col2:
        occupation = st.text_input("Occupation", value="Self_Emp")
        num_30_59 = st.number_input("NumberOfTime30-59DaysPastDueNotWorse", min_value=0.0, value=0.0)
        debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.5)
        num_open_loans = st.number_input("NumberOfOpenCreditLinesAndLoans", min_value=0.0, value=5.0)
        num_90_late = st.number_input("NumberOfTimes90DaysLate", min_value=0.0, value=0.0)
        num_real_estate = st.number_input("NumberRealEstateLoansOrLines", min_value=0.0, value=1.0)
        num_60_89 = st.number_input("NumberOfTime60-89DaysPastDueNotWorse", min_value=0.0, value=0.0)
        dependents = st.text_input("Number Of Dependents", value="2")

    submitted = st.form_submit_button("Predict")

if submitted:
    sample = {
        "NPA Status": npa_status,
        "RevolvingUtilizationOfUnsecuredLines": ruul,
        "age": age,
        "Gender": gender,
        "Region": region,
        "MonthlyIncome": monthly_income,
        "Rented_OwnHouse": rented_ownhouse,
        "Occupation": occupation,
        "Education": education,
        "NumberOfTime30-59DaysPastDueNotWorse": num_30_59,
        "DebtRatio": debt_ratio,
        "NumberOfOpenCreditLinesAndLoans": num_open_loans,
        "NumberOfTimes90DaysLate": num_90_late,
        "NumberRealEstateLoansOrLines": num_real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": num_60_89,
        "NumberOfDependents": dependents
    }

    unseen_df = pd.DataFrame([sample])
    result = predict_unseen(unseen_df, deploy_dict)

    pred_label = result["Predicted_Label"].iloc[0]
    prob_good = float(result["Prob_Good"].iloc[0])

    st.subheader("Prediction Result")
    st.write(f"**Predicted Credit Status:** {pred_label}")
    st.write(f"**Probability of Good:** {prob_good:.4f}")
