
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Model" / "loan_model.pkl"

if not MODEL_PATH.exists():
    st.error("Loan model not found.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Error loading model.")
    st.exception(e)
    st.stop()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef7ff, #f8fbff);
}

.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #075985;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 19px;
    color: #475569;
    margin-bottom: 25px;
}

.section-title {
    color: #0f172a;
    font-size: 25px;
    font-weight: 700;
}

.card {
    padding: 20px;
    border-radius: 18px;
    background: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.approved {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border: 2px solid #22c55e;
    text-align: center;
}

.rejected {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    text-align: center;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.result-subtitle {
    font-size: 18px;
    color: #334155;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #64748b;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>🏦 Loan Approval Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>🤖 AI Powered Banking Loan Prediction Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown("---")

with st.sidebar:
    st.title("🏦 Loan AI")

    st.info(
        "This AI system predicts whether a loan application "
        "is likely to be approved or rejected."
    )

    st.markdown("### 🤖 Models")
    st.write("• Logistic Regression")
    st.write("• Random Forest")
    st.write("• Decision Tree")

    st.markdown("---")

    st.markdown("### 👨‍💻 Developer")
    st.write("**Anand Shukla**")
    st.write("Software Engineer & Web Developer")
    st.write("B.Tech CSE – AKTU")
    st.write("AIML Summer Internship 2026")
    st.write("MNNIT Allahabad")

st.markdown(
    "<div class='section-title'>📋 Applicant Information</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

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

with col2:
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

with col3:
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

st.markdown("</div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0,
        value=360,
        step=10
    )

with col5:
    credit_history = st.selectbox(
        "Credit History",
        [0, 1],
        format_func=lambda x:
        "Good Credit History" if x == 1 else "Poor Credit History"
    )

with col6:
    total_income = applicant_income + coapplicant_income

    st.metric(
        "💰 Total Income",
        f"{total_income:,.0f}"
    )

st.markdown("---")

income_col, loan_col = st.columns(2)

with income_col:
    st.markdown("### 💰 Income Overview")

    st.bar_chart(
        {
            "Income": {
                "Applicant": applicant_income,
                "Co-applicant": coapplicant_income
            }
        },
        height=280
    )

with loan_col:
    st.markdown("### 🏦 Loan Overview")

    st.bar_chart(
        {
            "Amount": {
                "Loan Amount": loan_amount,
                "Total Income": total_income
            }
        },
        height=280
    )

st.markdown("---")

predict = st.button(
    "🔍 PREDICT LOAN STATUS",
    use_container_width=True,
    type="primary"
)

if predict:

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

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]

            if len(probabilities) == 2:
                rejected_probability = probabilities[0] * 100
                approved_probability = probabilities[1] * 100
            else:
                approved_probability = 100 if prediction == 1 else 0
                rejected_probability = 100 - approved_probability
        else:
            approved_probability = 100 if prediction == 1 else 0
            rejected_probability = 100 - approved_probability

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        if prediction == 1:

            st.markdown(
                f"""
                <div class='approved'>
                    <div class='result-title'>✅ LOAN APPROVED</div>
                    <div class='result-subtitle'>
                        Congratulations! Your loan application is predicted
                        to be approved.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class='rejected'>
                    <div class='result-title'>❌ LOAN REJECTED</div>
                    <div class='result-subtitle'>
                        The AI model predicts that the loan application
                        may not be approved.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Prediction Probability")

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:
            st.metric(
                "✅ Approval Probability",
                f"{approved_probability:.2f}%"
            )

        with probability_col2:
            st.metric(
                "❌ Rejection Probability",
                f"{rejected_probability:.2f}%"
            )

        st.progress(
            int(round(approved_probability))
        )

        st.markdown("### 📈 AI Prediction Graph")

        st.bar_chart(
            {
                "Probability (%)": {
                    "Loan Approved": approved_probability,
                    "Loan Rejected": rejected_probability
                }
            },
            height=350
        )

        st.markdown("### 📋 Application Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric(
                "Applicant Income",
                f"{applicant_income:,.0f}"
            )

        with summary_col2:
            st.metric(
                "Co-applicant Income",
                f"{coapplicant_income:,.0f}"
            )

        with summary_col3:
            st.metric(
                "Loan Amount",
                f"{loan_amount:,.0f}"
            )

        with summary_col4:
            st.metric(
                "Credit History",
                "Good" if credit_history == 1 else "Poor"
            )

        st.markdown("---")

        if prediction == 1:
            st.success(
                f"🎉 AI Confidence: {approved_probability:.2f}%"
            )
        else:
            st.warning(
                f"⚠️ AI Confidence: {rejected_probability:.2f}%"
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        <h4>🏦 Loan Approval Prediction System</h4>
        <p>Developed by <b>Anand Shukla</b></p>
        <p>AIML Summer Internship 2026 | MNNIT Allahabad</p>
    </div>
    """,
    unsafe_allow_html=True
)

