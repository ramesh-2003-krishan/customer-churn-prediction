import streamlit as st
import pandas as pd
import joblib

# Load model and columns
try:
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    st.error(f"Error loading model or model columns. Please run 'python main.py' first to train the model. Details: {e}")
    st.stop()

# Configure page metadata
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide"
)

# Apply some custom CSS styling for premium feel
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
    }
    .header-style {
        text-align: center;
        color: #1e3d59;
        font-weight: 800;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-style'>🛡️ Telco Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

st.subheader("Enter Customer Details")
st.markdown("Use the options below to enter the customer's demographics, subscriptions, and billing preferences.")

# Create the form for inputs
with st.form("churn_input_form"):
    # Split layout into 3 main columns
    col_demo, col_services, col_account = st.columns(3)
    
    # 1. Demographics
    with col_demo:
        st.markdown("### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        
    # 2. Services
    with col_services:
        st.markdown("### 🔌 Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        
        # Sub-services (only make sense if Internet Service != "No")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    # 3. Account / Contract information
    with col_account:
        st.markdown("### 💳 Account & Billing")
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=10.0)

    # Submit button
    submitted = st.form_submit_button("Predict Churn Outcome")

# If prediction form submitted
if submitted:
    # 1. Initialize input row matching model columns with 0
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0
    
    # 2. Map Numerical Columns
    if "Tenure Months" in input_data.columns:
        input_data["Tenure Months"] = tenure
    if "Monthly Charges" in input_data.columns:
        input_data["Monthly Charges"] = monthly_charges
    if "Total Charges" in input_data.columns:
        input_data["Total Charges"] = total_charges

    # 3. Map Categorical Columns Dynamically using "FeatureName_SelectedValue" matching
    cat_mapping = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method
    }
    
    for feature, selected_value in cat_mapping.items():
        col_name = f"{feature}_{selected_value}"
        if col_name in input_data.columns:
            input_data[col_name] = 1
            
    # 4. Predict probability and outcome
    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        churn_prob = probabilities[1]
    except Exception as ex:
        st.error(f"Error during prediction. Please verify that data matches expectations. Details: {ex}")
        st.stop()

    # 5. Display Prediction Outcome neatly
    st.write("---")
    st.subheader("📋 Prediction Findings")
    
    col_status, col_probability = st.columns(2)
    
    with col_status:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-card" style="background-color: #fce8e6; border: 2px solid #ea4335;">
                    <h3 style="color: #c5221f; margin: 0;">⚠️ Customer Likely to Churn</h3>
                    <p style="color: #601e1c; margin-top: 10px; font-weight: 500;">
                        The customer exhibits characteristics typically associated with cancellation. Immediate proactive outreach recommended.
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card" style="background-color: #e6f4ea; border: 2px solid #137333;">
                    <h3 style="color: #137333; margin: 0;">✅ Customer Likely to Stay</h3>
                    <p style="color: #0d652d; margin-top: 10px; font-weight: 500;">
                        The customer exhibits characteristics typical of loyal users. Retain standard relationship protocols.
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    with col_probability:
        st.write("")
        st.metric(
            label="Confidence (Predicted Probability of Churn)", 
            value=f"{churn_prob:.1%}",
            delta=f"{churn_prob - 0.50:+.1%}" if prediction == 1 else f"{churn_prob - 0.50:+.1%}",
            delta_color="inverse"
        )
        st.progress(float(churn_prob))
        st.caption(
            "Confidence shows the probability of Churn (Yes classification at threshold > 50%)."
        )