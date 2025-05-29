import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction")
st.write("Fill in the customer details to predict if they are likely to churn.")

# Simplified Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.checkbox("Senior Citizen")  # Boolean input
Partner = st.checkbox("Has Partner")
Dependents = st.checkbox("Has Dependents")
tenure = st.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.checkbox("Has Phone Service")
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.checkbox("Use Paperless Billing")
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Prepare and transform input for the model
input_data = {
    'gender': gender,
    'SeniorCitizen': 1 if SeniorCitizen else 0,
    'Partner': 'Yes' if Partner else 'No',
    'Dependents': 'Yes' if Dependents else 'No',
    'tenure': tenure,
    'PhoneService': 'Yes' if PhoneService else 'No',
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': 'Yes' if PaperlessBilling else 'No',
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_df = pd.DataFrame([input_data])

# Apply encoders
for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

# Predict
if st.button("🔍 Predict"):
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)

    label = "⚠️ Churn" if prediction[0] == 1 else "✅ No Churn"
    prob = prediction_proba[0][prediction[0]] * 100

    st.subheader("Prediction Result")
    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{prob:.2f}%**")

    st.markdown("### Prediction Probabilities")
    st.bar_chart({
        "No Churn": [prediction_proba[0][0]],
        "Churn": [prediction_proba[0][1]]
    })
