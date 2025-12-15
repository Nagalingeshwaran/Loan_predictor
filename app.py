import streamlit as st
import joblib
import numpy as np

# Load the trained model
clf = joblib.load("loan_approval_model.pkl")

# Streamlit UI for user input
st.title("Loan Approval Prediction")

# Collecting all 13 inputs
self_employed_no = st.radio("Self Employed?", ["No", "Yes"]) == "No"
self_employed_yes = not self_employed_no
education_graduate = st.radio("Education Level", ["Graduate", "Not Graduate"]) == "Graduate"
education_not_graduate = not education_graduate

no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=1)
income_annum = st.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
loan_term = st.number_input("Loan Term (in months)", min_value=1, value=36)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)

residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=100000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=50000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=30000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=20000)

# Prepare input array (Ensure all 13 features are included)
input_features = np.array([
    self_employed_no, self_employed_yes, 
    education_graduate, education_not_graduate,
    no_of_dependents, income_annum, loan_amount, loan_term, 
    cibil_score, residential_assets_value, commercial_assets_value, 
    luxury_assets_value, bank_asset_value
]).reshape(1, -1)

# Prediction
if st.button("Predict Loan Approval"):
    prediction = clf.predict(input_features)
    st.write(f"Loan Approval Status: {'Approved' if prediction[0] == 0 else 'Not Approved'}")
