
import streamlit as st
import pandas as pd
import joblib

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

# Input fields for features
age = st.slider('Age', 15, 70, 25)
systolic_bp = st.slider('SystolicBP', 70, 160, 120)
diastolic_bp = st.slider('DiastolicBP', 40, 110, 80)
bs = st.number_input('Blood Sugar (BS)', min_value=1.0, max_value=20.0, value=7.0, step=0.1)
body_temp = st.number_input('BodyTemp (Celsius)', min_value=35.0, max_value=42.0, value=37.0, step=0.1)
heart_rate = st.slider('HeartRate', 60, 120, 80)

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'Age': age,
    'SystolicBP': systolic_bp,
    'DiastolicBP': diastolic_bp,
    'BS': bs,
    'BodyTemp': body_temp,
    'HeartRate': heart_rate
}])

if st.button('Predict Risk'):
    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    # Map prediction to risk level string
    risk_levels = {0: 'low risk', 1: 'mid risk', 2: 'high risk'}
    predicted_risk = risk_levels[prediction[0]]

    st.subheader('Prediction Result:')
    st.write(f"Predicted Risk Level: **{predicted_risk}**")
    
    st.subheader('Probabilities:')
    prob_df = pd.DataFrame({
        'Risk Level': ['low risk', 'mid risk', 'high risk'],
        'Probability': prediction_proba[0]
    })
    st.dataframe(prob_df.set_index('Risk Level'))
