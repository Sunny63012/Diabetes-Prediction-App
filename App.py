# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:33:57 2026

@author: komma
"""

import numpy as np
import pickle
import streamlit as st
import pandas
import os
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))
#loaded_model = pickle.load(open('trained_model.sav', 'rb'))
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
def main():
    st.title("Diabetes Prediction Web App")
    #Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BP level')
    SkinThickness=st.text_input('Thickness value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI Index')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction value')
    Age=st.text_input('Age of person')
    
    diagnosis=""
    
    if st.button("Diabetes test button"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)

if __name__=='__main__':
    main()
