%%writefile streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import io

# Load the model and scaler
try:
    model = joblib.load('best_maternal_health_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title('Maternal Health Risk Prediction')
st.write('Enter the patient\"s physiological data to predict the maternal health risk level.')

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
        body_temp = (body_temp_input - 32) * 5/9 # Convert Fahrenheit to Celsius

    # Check if any validation failed
    if any(x is None for x in [age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]):
        st.stop()

    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Age': age,
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'BS': bs,
        'BodyTemp': body_temp,
        'HeartRate': heart_rate
    }])

    # Scale the input data
    scaled_data = scaler.transform(input_data)
    scaled_df = pd.DataFrame(scaled_data, columns=input_data.columns)

    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    # Map prediction to risk level string
    risk_levels = {0: 'low risk', 1: 'mid risk', 2: 'high risk'}
    predicted_risk = risk_levels[prediction[0]]

    st.subheader('Prediction Result:')
    st.markdown(f"Predicted Risk Level: **{predicted_risk.upper()}**")

    st.subheader('Detailed Prediction Information:')
    
    # Prepare results for the table
    results_data = {
        'Feature': input_data.columns.tolist() + ['Predicted Risk'] + [f'Probability ({rl})' for rl in risk_levels.values()],
        'Original Value': input_data.iloc[0].tolist() + ['-'] + ['-' for _ in risk_levels.values()],
        'Scaled Value': scaled_df.iloc[0].tolist() + ['-'] + ['-' for _ in risk_levels.values()]
    }
    # Update 'Original Value' and 'Scaled Value' for 'Predicted Risk' and Probabilities for display
    results_data['Original Value'][-len(risk_levels):] = ['-' for _ in risk_levels.values()]
    results_data['Scaled Value'][-len(risk_levels):] = ['-' for _ in risk_levels.values()]

    # Add predicted risk and probabilities to the table
    results_data['Original Value'][len(input_data.columns)] = predicted_risk.upper()
    results_data['Scaled Value'][len(input_data.columns)] = predicted_risk.upper() # Re-using scaled column for display
    for i, (level, prob) in enumerate(zip(risk_levels.values(), prediction_proba[0])):
        results_data['Original Value'][len(input_data.columns) + 1 + i] = f'{prob:.4f}'
        results_data['Scaled Value'][len(input_data.columns) + 1 + i] = f'{prob:.4f}' # Re-using scaled column for display

    results_df_display = pd.DataFrame(results_data)
    st.dataframe(results_df_display)

    # Add export functionality
    csv_buffer = io.StringIO()
    results_df_display.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Prediction Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="maternal_health_prediction_results.csv",
        mime="text/csv"
    )
