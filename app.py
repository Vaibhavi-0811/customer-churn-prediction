import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("🔮 Customer Churn Prediction")

st.write(
    "Enter customer details below to predict churn probability "
    "and identify the customer's risk level."
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0
    )


with col2:

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col3:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [
            "No",
            "Yes"
        ]
    )

    partner = st.selectbox(
        "Has Partner?",
        [
            "No",
            "Yes"
        ]
    )

    dependents = st.selectbox(
        "Has Dependents?",
        [
            "No",
            "Yes"
        ]
    )


# ============================================================
# SERVICES
# ============================================================

st.header("📱 Services")

col1, col2, col3, col4 = st.columns(4)


with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes"]
    )


with col2:

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )


with col4:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )


paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)


st.divider()


# ============================================================
# CREATE CUSTOMER DATA
# ============================================================

input_data = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [
        1 if senior_citizen == "Yes" else 0
    ],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone_service],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet_service],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless_billing],

    "PaymentMethod": [payment_method],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]

})


# ============================================================
# ENCODING
# ============================================================

input_encoded = pd.get_dummies(
    input_data,
    drop_first=True
)

input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)


# ============================================================
# SCALING
# ============================================================

input_scaled = scaler.transform(input_encoded)


# ============================================================
# PREDICTION
# ============================================================

st.header("🔮 Prediction")


if st.button("🚀 Predict Customer Risk", use_container_width=True):

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    prediction = model.predict(
        input_scaled
    )[0]


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability > 0.60:

        risk = "High Risk"

    elif probability >= 0.30:

        risk = "Medium Risk"

    else:

        risk = "Low Risk"


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    with col2:

        st.metric(
            "Prediction",
            prediction
        )


    with col3:

        st.metric(
            "Risk Level",
            risk
        )


    st.progress(float(probability))


    # ========================================================
    # REVENUE EXPOSURE
    # ========================================================

    if risk == "High Risk":

        revenue_exposure = monthly_charges

    else:

        revenue_exposure = 0


    st.subheader("💰 Business Value")

    st.metric(
        "Monthly Revenue Exposure",
        f"${revenue_exposure:,.2f}"
    )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.subheader("👤 Customer Summary")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Tenure",
            f"{tenure} months"
        )


    with col2:

        st.metric(
            "Contract",
            contract
        )


    with col3:

        st.metric(
            "Internet",
            internet_service
        )


    with col4:

        st.metric(
            "Monthly Charges",
            f"${monthly_charges:,.2f}"
        )


    # ========================================================
    # INSIGHT
    # ========================================================

    st.subheader("💡 Customer Insight")


    if risk == "High Risk":

        st.warning(
            "This customer has a relatively high predicted "
            "churn probability. Retention attention may be useful."
        )

    elif risk == "Medium Risk":

        st.info(
            "This customer has a moderate predicted churn "
            "probability and can be monitored."
        )

    else:

        st.success(
            "This customer currently has a lower predicted "
            "churn probability."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction System • "
    "Python • Pandas • Scikit-learn • Streamlit"
)