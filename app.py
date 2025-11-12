import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
try:
    model = joblib.load('best_maternal_health_risk_model (5).pkl')
    scaler = joblib.load('scaler (5).pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title('Maternal Health Risk Prediction')
st.write('Enter the patient\'s physiological data to predict the maternal health risk level.')

# Input fields for features as text boxes
age_str = st.text_input('Age', value='25')
systolic_bp_str = st.text_input('SystolicBP', value='120')
diastolic_bp_str = st.text_input('DiastolicBP', value='80')
bs_str = st.text_input('Blood Sugar (BS)', value='7.0')
body_temp_str = st.text_input('BodyTemp', value='98.6')
body_temp_unit = st.radio("Body Temperature Unit", ('Fahrenheit', 'Celsius'))
heart_rate_str = st.text_input('HeartRate', value='80')

# Function to validate and convert inputs
def validate_and_convert(input_str, type_func, feature_name):
    try:
        return type_func(input_str)
    except ValueError:
        st.error(f"Invalid input for {feature_name}. Please enter a valid number.")
        return None

if st.button('Predict Risk'):
    # Convert string inputs to numeric, with validation
    age = validate_and_convert(age_str, int, 'Age')
    systolic_bp = validate_and_convert(systolic_bp_str, int, 'SystolicBP')
    diastolic_bp = validate_and_convert(diastolic_bp_str, int, 'DiastolicBP')
    bs = validate_and_convert(bs_str, float, 'Blood Sugar (BS)')
    heart_rate = validate_and_convert(heart_rate_str, int, 'HeartRate')

    # Handle body temperature conversion
    body_temp_input = validate_and_convert(body_temp_str, float, 'BodyTemp')
    if body_temp_input is None:
        st.stop()
    
    body_temp = body_temp_input
    if body_temp_unit == 'Fahrenheit':
        body_temp = (body_temp_input - 32) * 5/9  # Convert to Celsius if Fahrenheit

    # Check if all conversions were successful
    if None in [age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]:
        st.stop()

    # Prepare features for prediction
    features = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction_numeric = model.predict(scaled_features)[0]

    # Map numeric prediction to risk levels
    # Assuming the 'Level' column was encoded as 0, 1, 2 for 'low', 'mid', 'high' risk
    risk_levels = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}
    predicted_risk = risk_levels.get(prediction_numeric, 'Unknown Risk Level')

    st.success(f"Predicted Maternal Health Risk: **{predicted_risk}**")
