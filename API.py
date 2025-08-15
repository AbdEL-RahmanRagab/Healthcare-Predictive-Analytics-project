import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Load the model with error handling
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please upload the model file.")
    st.stop()  # Stop further execution if the model is missing
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Page title
st.title("🔍 Medical Classification Model")
st.write("Enter patient data to classify the medical condition using the model.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ['Female', 'Male'])
gender_value = 1 if gender == 'Male' else 0

blood_type = st.selectbox("Blood Type", ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'])
blood_type_dict = {
    'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
    'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
}
blood_type_value = blood_type_dict[blood_type]

condition = st.selectbox("Medical Condition", ['Arthritis', 'Asthma', 'Cancer', 'Diabetes', 'Hypertension', 'Obesity'])
condition_dict = {
    'Arthritis': 0, 'Asthma': 1, 'Cancer': 2,
    'Diabetes': 3, 'Hypertension': 4, 'Obesity': 5
}
condition_value = condition_dict[condition]

billing = st.number_input("Billing Amount", value=10000.0, min_value=0.0)
room_number = st.number_input("Room Number", value=200)

admission_type = st.selectbox("Admission Type", ['Elective', 'Urgent', 'Emergency'])
admission_type_dict = {'Elective': 0, 'Urgent': 1, 'Emergency': 2}
admission_type_value = admission_type_dict[admission_type]

medication = st.selectbox("Medication Type", ['Lipitor', 'Ibuprofen', 'Aspirin', 'Paracetamol', 'Penicillin'])
medication_dict = {
    'Lipitor': 0, 'Ibuprofen': 1,
    'Aspirin': 2, 'Paracetamol': 3,
    'Penicillin': 4
}
medication_value = medication_dict[medication]

if st.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([
        age, gender_value, blood_type_value, condition_value,
        billing, room_number, admission_type_value, medication_value
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Show result
    if prediction == 0:
        st.success("🔔 Prediction: Classified as Negative (No abnormal condition detected).")
        st.info("The patient does not show any signs of a medical issue according to the model.")
    else:
        st.success("🔔 Prediction: Classified as Positive (Potential medical issue detected).")
        st.info("The patient might have a medical condition that requires further investigation.")

    # Save input data + prediction + timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_dict = {
        'Timestamp': [timestamp],
        'Age': [int(age)],
        'Gender': [gender],
        'Blood_Type': [blood_type],
        'Condition': [condition],
        'Billing_Amount': [float(billing)],
        'Room_Number': [int(room_number)],
        'Admission_Type': [admission_type],
        'Medication': [medication],
        'Prediction': ['Positive' if prediction == 1 else 'Negative']
    }
    new_row = pd.DataFrame(data_dict, columns=[
        'Timestamp', 'Age', 'Gender', 'Blood_Type', 'Condition',
        'Billing_Amount', 'Room_Number', 'Admission_Type',
        'Medication', 'Prediction'
    ])

    file_name = 'data_log.csv'
    columns = ['Timestamp', 'Age', 'Gender', 'Blood_Type', 'Condition', 'Billing_Amount',
               'Room_Number', 'Admission_Type', 'Medication', 'Prediction']

    # Check if the file exists to determine if the header is needed
    if os.path.exists(file_name):
        new_row.to_csv(file_name, mode='a', header=False, index=False)
    else:
        new_row.to_csv(file_name, mode='w', header=True, index=False)
    st.info("✅ Patient data saved successfully.")

    # عرض البيانات المدخلة بشكل منظم
    st.subheader("📋 Entered Patient Data")
    st.dataframe(new_row)

