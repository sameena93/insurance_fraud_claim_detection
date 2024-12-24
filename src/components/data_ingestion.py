import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')



def clean_data(df):
    """
    Handle the missing values"""
    
    df['Claim_Amount'] = df['Claim_Amount'].fillna(0)
    logging.info("Clean data done.")
    return df


def feature_engineering(df):
    """
    Create new features to enhance fraud detection"""
    logging.info("Feature Engineering started....")
    # Create a flag for high claim amounts
    df['High_Claim'] = (df['Claim_Amount'] > 50000).astype(int)

    # Create a feature for suspicious timing
    df['Suspicious_Timing'] = (df['Time_Between_Claims'] <= 7).astype(int)



    """Normalizes Claim Amount by Age:
        Insurance claims can vary significantly across age groups.
        For instance, older individuals may have higher medical claims,
        while younger individuals may have lower claims. By creating the Cost per Age ratio, 
        you normalize the claim amount relative to age. 
        This allows the model to better understand the relationship between a person's age 
        and their typical claim behavior."""
    
    df['Cost_per_Age'] = df['Claim_Amount'] / (df['Age'] +1)

    # Anomaly detection for Claim Amount using Z-score and IQR
    df['Claim_Amount_Z'] = zscore(df['Claim_Amount'])


    #Calculate IQR for claim AMount
    Q1= df['Claim_Amount'].quantile(0.25)
    Q3= df['Claim_Amount'].quantile(0.75)
    IQR = Q3 - Q1

    #Identify outlier based on IQR
    df['outlier_iqr'] = np.where(
        (df['Claim_Amount'] < (Q1 - 1.5 * IQR)) |
        (df['Claim_Amount'] > (Q3 + 1.5 * IQR)) ,1,0
    )

    df['outlier_Z_score'] = np.where((df['Claim_Amount_Z'] > 3) | (df['Claim_Amount_Z'] < -3),1,0)

    # Combine the flags for both methods
    df['Outlier_Flag'] = df['outlier_Z_score'] | df['outlier_iqr']

    # MArk rows ad yes for outlier and No otherwise
    df['Is_Outlier'] = np.where(df['Outlier_Flag'] == 1, "Yes", "No")

    # Group by 'Patient_ID' and calculate the number of claims and the gap between claims
    patient_claims = df.groupby('Patient_ID').agg(
    claim_count = ('Claim_Amount', 'count'),
    avg_gap = ('Time_Between_Claims', 'mean')).reset_index()

    df = df.merge(patient_claims, on='Patient_ID', how='left')
    logging.info('Feature Engineering finished.')
    return df



def preprocess_input_data(df):
    """
    Feature encoding and scaling for the input data."""

    logging.info('Preprocessing data started.....')
    # define categorical and numerical columns
    categorical_columns = ['High_Claims','Suspicious_Timing', 'Specializion']

    numerical_columns = ['Claim_Amount','Age','Cost_per_Age','claim_count','avg_gap','Reputation_Score']

   
    #Create column transformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num',StandardScaler(), numerical_columns),
            ('cat',OneHotEncoder(sparse=False), categorical_columns)
        ]
    )

    processed_data = preprocessor.fit_transform(df)

    logging.info("preprocessing data finished.")
    return processed_data, preprocessor




