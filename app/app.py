import streamlit as st
import numpy as np
import pickle
import pandas as pd


with open("../model/credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.write("Predict whether a loan applicant is **High Risk** or **Low Risk**")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (months)", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])


Total_Income = ApplicantIncome + CoapplicantIncome
Debt_Income_Ratio = LoanAmount / Total_Income if Total_Income > 0 else 0


input_data = {
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Total_Income": Total_Income,
    "Debt_Income_Ratio": Debt_Income_Ratio,
}


cat_features = {
    "Gender_Male": 1 if Gender == "Male" else 0,
    "Married_Yes": 1 if Married == "Yes" else 0,
    "Dependents_1": 1 if Dependents == "1" else 0,
    "Dependents_2": 1 if Dependents == "2" else 0,
    "Dependents_3+": 1 if Dependents == "3+" else 0,
    "Education_Not Graduate": 1 if Education == "Not Graduate" else 0,
    "Self_Employed_Yes": 1 if Self_Employed == "Yes" else 0,
    "Property_Area_Semiurban": 1 if Property_Area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if Property_Area == "Urban" else 0,
}

input_data.update(cat_features)

input_df = pd.DataFrame([input_data])

if st.button("Predict Risk"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk Applicant (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk Applicant (Probability: {1 - probability:.2f})")
