
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Model" / "loan_model.pkl"

if not MODEL_PATH.exists():
    st.error("❌ Loan model not found.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Error loading model.")
    st.exception(e)
    st.stop()

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #0E76A8;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 20px;
}
.footer {
    text-align: center;
    padding: 20px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='main-title'>🏦 Loan Approval Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>AI Powered Banking Loan Prediction</p>",
    unsafe_allow_html=True
)

st.markdown("---")

st.sidebar.title("📌 Project Information")

st.sidebar.info("""
This project predicts whether a loan application will be approved or rejected using Machine Learning.

Models Used:
✅ Logistic Regression
✅ Random Forest
✅ Decision Tree
""")

st.sidebar.markdown("""
### 👨‍💻 Developer

**Anand Shukla**

Software Engineer & Web Developer

B.Tech Engineering – AKTU

AIML Summer Internship 2026

MNNIT Allahabad, Prayagraj
""")

st.markdown("### 📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

with col2:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=1500,
        step=100
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=120,
        step=10
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0,
        value=360,
        step=10
    )

    credit_history = st.selectbox(
        "Credit History",
        [0, 1],
        format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)"
    )

total_income = applicant_income + coapplicant_income

st.metric(
    "💰 Total Applicant Income",
    f"{total_income:,.0f}"
)

st.markdown("---")

if st.button("🔍 Predict Loan Status", use_container_width=True):

    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    dependents_encoded = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3+": 3
    }[dependents]

    property_area_encoded = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]

    features = np.array([
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area_encoded,
        total_income
    ]).reshape(1, -1)

    try:
        prediction = model.predict(features)[0]

        st.markdown("---")

        if prediction == 1:
            st.success("✅ Congratulations! Your Loan is Approved.")
            st.balloons()
            st.metric("Prediction Status", "APPROVED")
        else:
            st.error("❌ Sorry! Your Loan is Rejected.")
            st.metric("Prediction Status", "REJECTED")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)

st.markdown("---")

st.markdown("""
<div class="footer">
<h4>Developed by Anand Shukla</h4>
<p>AIML Summer Internship 2026</p>
<p>MNNIT Allahabad, Prayagraj</p>
</div>
""", unsafe_allow_html=True)
