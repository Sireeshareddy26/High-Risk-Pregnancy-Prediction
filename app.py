import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os

# ---------------- Load Model ---------------- #
try:
    model = joblib.load('best_maternal_health_risk_model (5).pkl')
    scaler = joblib.load('scaler (5).pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# ---------------- Page Title ---------------- #
st.title('Maternal Health Risk Prediction')
st.write("Enter the patient's physiological data to predict maternal health risk.")

# ---------------- Input Fields ---------------- #
age_str = st.text_input('Age', value='25')
systolic_bp_str = st.text_input('SystolicBP', value='120')
diastolic_bp_str = st.text_input('DiastolicBP', value='80')
bs_str = st.text_input('Blood Sugar (BS)', value='7.0')
body_temp_str = st.text_input('BodyTemp', value='98.6')
body_temp_unit = st.radio("Body Temperature Unit", ('Fahrenheit', 'Celsius'))
heart_rate_str = st.text_input('HeartRate', value='80')

# ---------------- Helper Functions ---------------- #
def validate_and_convert(input_str, type_func, feature_name):
    try:
        return type_func(input_str)
    except ValueError:
        st.error(f"Invalid input for {feature_name}.")
        return None

DATA_FILE = "daily_entries.csv"

def save_entry(data):
    df_new = pd.DataFrame([data])

    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(DATA_FILE, index=False)

def load_last_24_hours():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    cutoff = datetime.now() - timedelta(hours=24)
    df_filtered = df[df['Timestamp'] >= cutoff]

    df_filtered.to_csv(DATA_FILE, index=False)
    return df_filtered

# ---------------- Prediction ---------------- #
if st.button('Predict Risk'):

    age = validate_and_convert(age_str, int, 'Age')
    systolic_bp = validate_and_convert(systolic_bp_str, int, 'SystolicBP')
    diastolic_bp = validate_and_convert(diastolic_bp_str, int, 'DiastolicBP')
    bs = validate_and_convert(bs_str, float, 'Blood Sugar')
    heart_rate = validate_and_convert(heart_rate_str, int, 'HeartRate')

    body_temp_input = validate_and_convert(body_temp_str, float, 'BodyTemp')
    if body_temp_unit == 'Fahrenheit':
        body_temp = (body_temp_input - 32) * 5/9
    else:
        body_temp = body_temp_input

    if None in [age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]:
        st.stop()

    features = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                             columns=['Age','SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate'])

    scaled_features = scaler.transform(features)
    prediction_numeric = model.predict(scaled_features)[0]

    risk_levels = {0: 'Low Risk', 1: 'Mid Risk', 2: 'High Risk'}
    predicted_risk = risk_levels.get(prediction_numeric, 'Unknown')

    st.success(f"Predicted Maternal Health Risk: **{predicted_risk}**")

    # Save prediction
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BloodSugar": bs,
        "BodyTemp(C)": round(body_temp,2),
        "HeartRate": heart_rate,
        "PredictedRisk": predicted_risk
    }

    save_entry(entry)

# ---------------- Daily Entries Section ---------------- #
st.markdown("---")
st.subheader("📊 Daily Entries (Last 24 Hours)")

df_entries = load_last_24_hours()

if not df_entries.empty:
    st.dataframe(df_entries, use_container_width=True)

    csv = df_entries.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="daily_maternal_health_entries.csv",
        mime='text/csv'
    )
else:
    st.info("No entries recorded in the last 24 hours.")
