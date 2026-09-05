import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("../Model/loan_model.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    color:#0E76A8;
}
.result-box{
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<h1 class='main-title'>🏦 Loan Approval Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>AI Powered Banking Loan Prediction</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.info(
"""
This project predicts whether a loan
application will be approved or rejected
using Machine Learning.

Models Used:
✅ Logistic Regression
✅ Random Forest
✅ Decision Tree
"""

"""Developer Details:
This project has been developed by Anand Shukla, a passionate Software Engineer and Web Developer. The project demonstrates his technical skills and practical experience in modern web development. He is currently pursuing Engineering from AKTU and is also the founder of Rudra Digital Marketing Company of India. His main focus is on modern web technologies and full-stack development."""""
)

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=1500
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=120
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0,
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [0, 1]
    )

# -----------------------------
# Total Income
# -----------------------------
total_income = applicant_income + coapplicant_income

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Loan Status"):

    # Encoding

    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    education = 0 if education == "Graduate" else 1

    self_employed = 1 if self_employed == "Yes" else 0

    if dependents == "0":
        dependents = 0
    elif dependents == "1":
        dependents = 1
    elif dependents == "2":
        dependents = 2
    else:
        dependents = 3

    if property_area == "Rural":
        property_area = 0
    elif property_area == "Semiurban":
        property_area = 1
    else:
        property_area = 2

    features = np.array([
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        total_income
    ]).reshape(1, -1)

    prediction = model.predict(features)

    st.markdown("---")

    if prediction[0] == 1:

        st.success("✅ Congratulations! Loan Approved")

        st.balloons()

        st.metric(
            label="Prediction Status",
            value="Approved"
        )

    else:

        st.error("❌ Sorry! Loan Rejected")

        st.metric(
            label="Prediction Status",
            value="Rejected"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <center>
        <h4>Developed by Anand Shukla</h4>
        <p>AIML Summer Internship 2026</p>
        <p>MNNIT Allahabad, Prayagraj</p>
    </center>
    """,
    unsafe_allow_html=True
)