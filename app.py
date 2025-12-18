import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Prediction App")
st.write("Predict whether a loan will be **Approved or Rejected**")

# Load trained model
@st.cache_resource
def load_model():
    with open("loan_approval_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("✅ Model loaded successfully")

# ---------------- USER INPUTS ----------------
st.subheader("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Term (months)", [360, 240, 180, 120])
credit_history = st.selectbox("Credit History", [1.0, 0.0])

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# -------- ENCODING (Must match training) --------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

property_area_map = {
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
}
property_area = property_area_map[property_area]

# Feature array
features = np.array([[gender, married, education, self_employed,
                      applicant_income, coapplicant_income,
                      loan_amount, loan_term, credit_history,
                      property_area]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Machine Learning")
