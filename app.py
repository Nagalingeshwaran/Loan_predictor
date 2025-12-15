import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("loan_approval_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to check loan approval status")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding inputs (must match training encoding)
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0

dependents = 3 if dependents == "3+" else int(dependents)

property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_area_map[property_area]

# Prediction
if st.button("Predict Loan Status"):
    input_data = np.array([[gender, married, dependents, education,
                            self_employed, applicant_income, coapplicant_income,
                            loan_amount, loan_term, credit_history, property_area]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
