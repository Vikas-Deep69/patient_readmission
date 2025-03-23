import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained models
rf_model = joblib.load("rf_tuned_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")

# Load encoders and feature order
label_encoders = joblib.load("label_encoders.pkl")
label_encoders_2 = joblib.load("label_encoders_2.pkl")
feature_order = joblib.load("feature_columns.pkl")

st.title("🏥 Hospital Readmission Prediction App")

st.sidebar.header("📋 Enter Patient Information")

# User input fields
time_in_hospital = st.sidebar.number_input("Days in Hospital", min_value=1, max_value=30, step=1)
n_lab_procedures = st.sidebar.number_input("Number of Lab Procedures", min_value=0, max_value=100, step=1)
n_procedures = st.sidebar.number_input("Number of Procedures", min_value=0, max_value=10, step=1)
n_medications = st.sidebar.number_input("Number of Medications", min_value=0, max_value=50, step=1)
age = st.sidebar.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
glucose_test = st.sidebar.selectbox("Glucose Test", ["Yes", "No"])
A1Ctest = st.sidebar.selectbox("A1C Test", ["Yes", "No"])
change = st.sidebar.selectbox("Change in Medication", ["Yes", "No"])
diabetes_med = st.sidebar.selectbox("Diabetes Medication", ["Yes", "No"])

# Convert user input into a DataFrame
user_input_df = pd.DataFrame({
    "time_in_hospital": [time_in_hospital],
    "n_lab_procedures": [n_lab_procedures],
    "n_procedures": [n_procedures],
    "n_medications": [n_medications],
    "age": [age],
    "glucose_test": [glucose_test],
    "A1Ctest": [A1Ctest],
    "change": [change],
    "diabetes_med": [diabetes_med]
})

# Encode categorical features
for col in label_encoders:
    if col in user_input_df:
        user_input_df[col] = user_input_df[col].apply(lambda x: x if x in label_encoders[col].classes_ else "Unknown")
        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")
        user_input_df[col] = label_encoders[col].transform(user_input_df[col])

# Ensure the feature order matches the training data
user_input_df = user_input_df.reindex(columns=feature_order, fill_value=0)

# Predict using the models
if st.sidebar.button("🔍 Predict Readmission"):
    rf_prediction = rf_model.predict(user_input_df)[0]
    rf_proba = rf_model.predict_proba(user_input_df)[0][1]

    xgb_prediction = xgb_model.predict(user_input_df)[0]
    xgb_proba = xgb_model.predict_proba(user_input_df)[0][1]

    lgbm_prediction = lgbm_model.predict(user_input_df)[0]
    lgbm_proba = lgbm_model.predict_proba(user_input_df)[0][1]

    # Display Results
    st.subheader("📊 Prediction Results")

    def format_prediction(pred, proba):
        if pred == 0:
            return f"🔴 **Likely to be Readmitted** (Probability: {proba:.2%})"
        else:
            return f"🟢 **Not Likely to be Readmitted** (Probability: {proba:.2%})"

    st.write("**Random Forest:**", format_prediction(rf_prediction, rf_proba))
    st.write("**XGBoost:**", format_prediction(xgb_prediction, xgb_proba))
    st.write("**LightGBM:**", format_prediction(lgbm_prediction, lgbm_proba))
    
    # Choose final prediction (majority vote)
    final_prediction = round((rf_prediction + xgb_prediction + lgbm_prediction) / 3)
    final_proba = (rf_proba + xgb_proba + lgbm_proba) / 3

    st.markdown("### 🏥 **Final Prediction:**")
    st.write(format_prediction(final_prediction, final_proba))
