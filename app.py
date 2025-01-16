import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def user_input_form():
    st.title("Health Insurance Claim Fraud Detection")

    # input fields for numerical colmns
    claim_amount = st.number_input("Claim Amount", min_value=0)
    age = st.number_input("Age", min_value=1, max_value = 100)
    cost_per_age = st.number_input('Cost Per Age', min_value =0)
    claim_count = st.number_input('Claim Count', min_value = 0)
    avg_gap = st.number_input("Average Gap", min_value=0)
    reputation_score = st.number_input("Reputation Score", min_value=0)

    # Input fields for categorical features
    high_claim= st.selectbox('High Claim', ['Yes', 'No'])
    suspicious_timing = st.selectbox("Suspicious Timing", ["Yes", "No"])
    medical_history = st.selectbox("Medical History", ["None", "Hypertension", "Diabetes", "Asthama"])
    specialization = st.selectbox("Specialization", ["Orthopedics", "Cardiology","Emergency Medicine","Dermatology"])


    # convert to dataframe
    input_date = {
        'Claim_Amount': claim_amount,
        'Age': age,
        'Cost_per_Age': cost_per_age,
        'claim_count': claim_count,
        'avg_gap': avg_gap,
        'Reputation_Score': reputation_score,
        'High_Claim': 1 if high_claim == "Yes" else 0,
        'Suspicious_Timing': 1 if suspicious_timing == "Yes" else 0,
        'Medical_History': medical_history,
        'Specialization': specialization
    }

    df = pd.DataFrame([input_date])

    return df

#Load the model 
import joblib
model = pickle.load(open("fraud_detection_xgb_model.pkl", "rb"))
lr_model = joblib.load(open('fraud_detection_lr_model.pkl','rb'))

preprocessor = pickle.load(open("preprocessor.pkl","rb"))
input_df = user_input_form()

# Prediction section
if st.button('Predict Fraudulent Claim'):
    

    
    input_df_processed = preprocessor.transform(input_df)
    
    # as preprocess use column transformer hence returning numpy array
    #convert the numpy array to a pandas dataframe and keep column names

    # input_df_processed = pd.DataFrame(input_df_processed, columns=preprocessor.get_feature_names_out())

    # prediction = model.predict(input_df_processed)
    lr_pred = lr_model.predict(input_df_processed)

    # st.write(f"Predcition: {'Fraudulent' if prediction[0] else 'Not Fraudulent'}")
    st.write(f"Predcition: {'Fraudulent' if lr_pred[0] else 'Not Fraudulent'}")



# streamlit run app.py

