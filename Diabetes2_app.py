# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:48:58 2021

@author: manis
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess the dataset
@st.cache
def load_data():
    # Load the dataset
    data = pd.read_csv('diabetes.csv')
    # Replace zeros in critical columns with the column mean
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_with_zeros:
        data[col] = data[col].replace(0, np.nan)
        data[col].fillna(data[col].mean(), inplace=True)
    return data

# Function to train the model
@st.cache
def train_model(data):
    # Split features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", accuracy)

    return rf_model

# Load and preprocess the dataset
st.title("Diabetes Prediction App")
st.write("Predict diabetes outcome based on input data using a Random Forest Classifier.")
data = load_data()

# Display dataset information
if st.checkbox("Show Dataset"):
    st.write(data)

# Train the model
model = train_model(data)

# User Input Form
st.header("Enter Patient Details:")
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin Level", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Create a DataFrame for input
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

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.subheader(f"Prediction: {result}")


