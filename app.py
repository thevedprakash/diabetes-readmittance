import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys, os
import joblib

# from list to dataframe

sys.path.append("src")
sys.path.append("models")

from predict import encode_predict
from encode_predict import *

st.set_page_config(page_title='Home',
                   page_icon='/home/ris/pythonProject/diabetes-readmittance/app/directory/logo.png',
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title('Hospital Readmission')

tab1, tab2, tab3 = st.tabs(["Personal Information", "Previous Admission Information", "Medical History"])


with tab1:
    st.header('Enter Your Personal Information')

    name = st.text_input('Name')

    race = st.selectbox(
        'Race',
        ('Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'))

    gender = st.selectbox(
        'Gender',
        ('Male', 'Female', 'Other'))

    age = st.selectbox(
        'Age Group',
        ('[70-80)', '[60-70)', '[50-60)', '[80-90)', '[40-50)', '[30-40)', '[90-100)', '[20-30)', '[10-20)', '[0-10)'))



with tab2:
    st.header('Previous Admission Information')

    admission_source_id = st.selectbox(
        'Admission Source',
        ('Transfer From Another Home Health Agency ',
         'Transfer from a hospital',
         'Transfer from another health care facility ',
         'Transfer from a Skilled Nursing Facility (SNF)',
         'Emergency Room',
         'Other'))

    admission_type_id = st.selectbox(
        'Admission Type',
        ('Emergency', 'Elective', 'Urgent', 'Other'))

    number_diagnoses = st.number_input(
        'Number of times Diabetes Detected Last Year', min_value=0, max_value=50, step=1
    )

    number_inpatient = st.number_input(
        'Number of Visits Last Year', min_value=0, max_value=50, step=1
    )

    time_in_hospital = st.number_input(
        'Period of Hospitalisation (Days)', min_value=1, max_value=15, step=1
    )

    number_emergency = st.number_input(
        'Number of Emergency Visits', min_value=0, max_value=50, step=1
    )

    number_outpatient = st.number_input(
        'Number of outpatient', min_value=0, max_value=50, step=1
    )

    discharge_disposition_id = st.selectbox(
        'Last Time Discharge Reason',
        ('Discharged to home',
         'Discharged/transferred to SNF',
         'Discharged/transferred to home with home health service',
         'Discharged/transferred to another short term hospital',
         'Discharged/transferred to another rehab fac including rehab units of a hospital .',
         'Expired',
         'Other'))

with tab3:
    st.header('Medical History')

    insulin = st.selectbox(
        'Insulin Level Change',
        ('No', 'Steady', 'Up', 'Down')
    )

    num_lab_procedures = st.number_input(
        'Number of Lab tests performed Last Year', min_value=0, max_value=150, step=1
    )

    num_procedures = st.number_input(
        'Number of test performed Last Year (Except Lab)', min_value=0, max_value=150, step=1
    )

    diabetesMed = st.selectbox(
        'Medicines of Diabetes Prescribed ?',
        ('Yes', 'No')
    )

    change = st.selectbox(
        'Medicines of Diabetes Changed ?',
        ('No', 'Ch')
    )

    num_medications = st.number_input(
        'Number of Medicines Prescribed Last Year', min_value=1, max_value=100, step=1
    )

    if st.button("SUBMIT"):
        features = {
            'race' : race,
            'gender' : gender,
            'age'    : age,
            'admission_type_id' : admission_type_id,
            'admission_source_id' : admission_source_id,
            'number_diagnoses' : number_diagnoses,
            'number_inpatient' : number_inpatient,
            'time_in_hospital' : time_in_hospital,
            'number_emergency' : number_emergency,
            'number_outpatient' : number_outpatient,
            'discharge_disposition_id' : discharge_disposition_id,
            'insulin' : insulin,
            'num_lab_procedures' : num_lab_procedures,
            'num_procedures' : num_procedures,
            'diabetesMed' : diabetesMed,
            'change'         : change,
            'num_medications' : num_medications}

        df = pd.DataFrame(features, index = [0])
        print(df)
        with open("models/Logistic_regression.pickle", 'rb') as handle:
            saved_model = pickle.load(handle)

        with open('models/transform_dict.pickle', 'rb') as handle:
            encoded_dict = pickle.load(handle)
        X = encode_predict(df, encoded_dict)

        prediction = saved_model.predict(X)
        readmittance_prob = np.array(['No', 'Yes',])
        st.write(readmittance_prob[prediction])
#--------------------------------------------------------------- personal information----------------------------------------------------------






#------------------------------------------Previous Admission information-----------------------------------------------





# ------------------------------------------medical history-------------------------------------------------------------------------




 #--------------------------------------------------Dataframe-----------------------------------------------------------------------

