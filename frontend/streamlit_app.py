import streamlit as st
import requests


API_URL = "https://customer-churn-api-wyv4.onrender.com/predict"


st.title("Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether they are likely to churn."
)


gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)


SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0,1]
)


Partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)


Dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)


tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)


PhoneService = st.selectbox(
    "Phone Service",
    ["Yes","No"]
)


MultipleLines = st.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


InternetService = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


MonthlyCharges = st.number_input(
    "Monthly Charges",
    value=50.0
)


TotalCharges = st.number_input(
    "Total Charges",
    value=500.0
)


Contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
if st.button("Predict Churn"):

    data = {

        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges

    }

    response = requests.post(
        API_URL,
        json=data
    )

    if response.status_code == 200:
        result = response.json()
        prediction = result["churn_prediction"]

        probability = result["churn_probability"]
        if prediction == 1:
            st.error(f"Customer likely to churn\nProbability: {probability:.2%}")

        else:
            st.success(f"Customer likely to stay\nProbability: {probability:.2%}")

    else:
        st.error("API error")