```python
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

# Project structure:
#
# Loan-Approval-Prediction/
# │
# ├── Model/
# │   └── loan_model.pkl
# │
# └── Streamlit_App/
#     ├── app.py
#     └── requirements.txt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Model" / "loan_model.pkl"

if not MODEL_PATH.exists():
    st.error("❌ Loan prediction model file was not found.")
    st.info(
        "Please make sure 'loan_model.pkl' exists inside the "
        "'Model' folder in your GitHub repository."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Error while loading the machine learning model.")
    st.exception(e)
    st.stop()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        color: #0E76A8;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #555555;
        font-size: 20px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        color: #0E76A8;
    }

    .footer {
        text-align: center;
        padding: 20px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div class='main-title'>🏦 Loan Approval Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Banking Loan Prediction</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Project Information")

st.sidebar.info(
    """
This project predicts whether a loan
application will be approved or rejected
using Machine Learning.

### Models Used

✅ Logistic Regression

✅ Random Forest

✅ Decision Tree
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### 👨‍💻 Developer

**Anand Shukla**

Software Engineer & Web Developer

B.Tech Engineering – AKTU

AIML Summer Internship 2026

MNNIT Allahabad, Prayagraj
"""
)

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    "<div class='section-title'>📋 Applicant Information</div>",
    unsafe_allow_html=True
)

st.write("Enter the applicant's details below to predict the loan status.")

col1, col2 = st.columns(2)

# ============================================================
# LEFT COLUMN
# ============================================================

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

# ============================================================
# RIGHT COLUMN
# ============================================================

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

# ============================================================
# TOTAL INCOME
# ============================================================

total_income = applicant_income + coapplicant_income

st.markdown("---")

st.metric(
    label="💰 Total Applicant Income",
    value=f"{total_income:,.0f}"
)

st.markdown("---")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def prepare_features():

    # Gender Encoding
    gender_encoded = 1 if gender == "Male" else 0

    # Married Encoding
    married_encoded = 1 if married == "Yes" else 0

    # Education Encoding
    education_encoded = 0 if education == "Graduate" else 1

    # Self Employed Encoding
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

    # Create feature array
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
        ],
        dtype=float
    ).reshape(1, -1)

    return features


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Predict Loan Status",
    use_container_width=True
):

    try:

        # Prepare input features
        features = prepare_features()

        # Make prediction
        prediction = model.predict(features)

        result = prediction[0]

        st.markdown("---")

        # ====================================================
        # APPROVED
        # ====================================================

        if result == 1:

            st.success(
                "✅ Congratulations! Your Loan is Approved."
            )

            st.balloons()

            st.metric(
                label="Prediction Status",
                value="APPROVED"
            )

        # ====================================================
        # REJECTED
        # ====================================================

        else:

            st.error(
                "❌ Sorry! Your Loan is Rejected."
            )

            st.metric(
                label="Prediction Status",
                value="REJECTED"
            )

    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <h4>Developed by Anand Shukla</h4>

    <p>AIML Summer Internship 2026</p>

    <p>MNNIT Allahabad, Prayagraj</p>

    <p>🏦 Loan Approval Prediction System</p>

    </div>
    """,
    unsafe_allow_html=True
)
```
