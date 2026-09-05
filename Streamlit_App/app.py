
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

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
# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path:
# Project/
# ├── Model/
# │   └── loan_model.pkl
# └── Streamlit_App/
#     └── app.py

MODEL_PATH = BASE_DIR / "Model" / "loan_model.pkl"

# Check whether model exists
if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# Load model
model = joblib.load(MODEL_PATH)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #0E76A8;
    }

    .result-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### 👨‍💻 Developer Details

    This project has been developed by
    **Anand Shukla**, a passionate Software
    Engineer and Web Developer.

    The project demonstrates technical skills
    and practical experience in Machine Learning
    and modern web development.

    He is currently pursuing Engineering from
    AKTU and is also the founder of
    **Rudra Digital Marketing Company of India**.

    His main focus is on modern web technologies
    and full-stack development.
    """
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

    # -------------------------
    # Encoding
    # -------------------------

    gender_encoded = 1 if gender == "Male" else 0

    married_encoded = 1 if married == "Yes" else 0

    education_encoded = 0 if education == "Graduate" else 1

    self_employed_encoded = 1 if self_employed == "Yes" else 0

    # Dependents Encoding
    if dependents == "0":
        dependents_encoded = 0
    elif dependents == "1":
        dependents_encoded = 1
    elif dependents == "2":
        dependents_encoded = 2
    else:
        dependents_encoded = 3

    # Property Area Encoding
    if property_area == "Rural":
        property_area_encoded = 0
    elif property_area == "Semiurban":
        property_area_encoded = 1
    else:
        property_area_encoded = 2

    # -------------------------
    # Feature Array
    # -------------------------
    features = np.array(
        [
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
        ]
    ).reshape(1, -1)

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(features)

    st.markdown("---")

    # -------------------------
    # Result
    # -------------------------
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
```
