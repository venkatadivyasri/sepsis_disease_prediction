import streamlit as st
import numpy as np
import catboost
from catboost import CatBoostClassifier

# Load the model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_sepsis_model.cbm")  # Ensure this file exists
    return model

model = load_model()

# Streamlit UI
st.title("🔬 Sepsis Prediction App")
st.write("Enter patient details to predict the likelihood of sepsis.")

# Input fields
HR = st.number_input("Heart Rate (HR)", min_value=0.0, max_value=200.0, value=90.0)
Resp = st.number_input("Respiratory Rate (Resp)", min_value=0.0, max_value=50.0, value=18.0)
Temp = st.number_input("Temperature (Temp)", min_value=90.0, max_value=110.0, value=99.0)
DBP = st.number_input("Diastolic Blood Pressure (DBP)", min_value=0.0, max_value=150.0, value=60.0)
SBP = st.number_input("Systolic Blood Pressure (SBP)", min_value=0.0, max_value=250.0, value=120.0)
MAP = st.number_input("Mean Arterial Pressure (MAP)", min_value=0.0, max_value=200.0, value=80.0)
O2Sat = st.number_input("Oxygen Saturation (O2Sat)", min_value=0.0, max_value=100.0, value=98.0)
Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0)
Gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

# Predict button
if st.button("Predict Sepsis"):
    # Convert inputs into numpy array
    input_data = np.array([[HR, Resp, Temp, DBP, SBP, MAP, O2Sat, Age, Gender]])

    # Ensure model gives the correct output
    prediction_prob = model.predict_proba(input_data)[:, 1]  # Probability of sepsis
    st.write(f"Sepsis Probability: {prediction_prob[0]:.4f}")  # Show probability for debugging

    # Apply threshold
    if prediction_prob[0] >= 0.5:
        st.error("⚠️ Sepsis Detected")
    else:
        st.success("✅ No Sepsis")
