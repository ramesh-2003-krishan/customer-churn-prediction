import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Prediction Dashboard")

st.write("Enter customer information")

# User inputs
tenure = st.slider("Tenure Months", 1, 72)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=120.0,
    help="Please enter a valid monthly charge"
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=9000.0,
    help="Please enter a valid total charge"
)

# Create empty dataframe
input_data = pd.DataFrame(
    columns=model_columns
)

# Add one row
input_data.loc[0] = 0

# Fill important values
if "Tenure Months" in input_data.columns:
    input_data["Tenure Months"] = tenure

if "Monthly Charges" in input_data.columns:
    input_data["Monthly Charges"] = monthly_charges

if "Total Charges" in input_data.columns:
    input_data["Total Charges"] = total_charges

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")