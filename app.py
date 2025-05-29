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

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction Model")

tab1, tab2 = st.tabs(["🔍 Predict One", "📁 Bulk Prediction"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.subheader("Single Customer Input")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.checkbox("Senior Citizen")
        Partner = st.checkbox("Has Partner")
        Dependents = st.checkbox("Has Dependents")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        PhoneService = st.checkbox("Has Phone Service")
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.checkbox("Use Paperless Billing")
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

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
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    if st.button("Predict"):
        pred = loaded_model.predict(input_df)[0]
        prob = loaded_model.predict_proba(input_df)[0][pred] * 100
        label = "⚠️ Churn" if pred == 1 else "✅ No Churn"

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {prob:.2f}%")
        st.bar_chart({"No Churn": [loaded_model.predict_proba(input_df)[0][0]],
                      "Churn": [loaded_model.predict_proba(input_df)[0][1]]})


with tab2:
    st.subheader("Upload CSV or Excel for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("🔍 Preview of uploaded data:")
            st.dataframe(df.head())

            # Drop customerID if exists
            if "customerID" in df.columns:
                df.drop(columns=["customerID"], inplace=True)

            # Encode categorical columns
            df_encoded = df.copy()
            for col, encoder in encoders.items():
                df_encoded[col] = encoder.transform(df_encoded[col])

            # Predict
            predictions = loaded_model.predict(df_encoded)
            probs = loaded_model.predict_proba(df_encoded)

            df['Prediction'] = ["Churn" if p == 1 else "No Churn" for p in predictions]
            df['Confidence (%)'] = [round(prob[p] * 100, 2) for prob, p in zip(probs, predictions)]

            st.success("✅ Prediction completed!")

            # 🔍 Show full results in a nice table
            st.subheader("📋 Prediction Results")
            st.dataframe(df)

            # 📥 Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv, "churn_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
