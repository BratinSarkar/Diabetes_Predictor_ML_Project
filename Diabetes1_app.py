# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:48:58 2021

@author: manis
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load trained model
@st.cache
def load_model():
    return pickle.load(open('random_forest_model.pkl', 'rb'))

# Prediction function
def predict_outcome(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction[0]

# App Interface
st.title("Diabetes Prediction App")
st.write("Predict diabetes outcome using patient data")

# User Input Form
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0)

# Convert inputs to DataFrame
input_features = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Prediction Button
if st.button("Predict"):
    outcome = predict_outcome(input_features)
    result = "Diabetic" if outcome == 1 else "Non-Diabetic"
    st.write(f"Prediction: {result}")

