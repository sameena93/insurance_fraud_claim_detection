import logging
import pandas as pd
from src.components.data_ingestion import feature_engineering, clean_data
from src.components.unsupervised_models import detect_anomalies_dbscan, detect_anomalies_isolation_forest
from src.components.preprocessing import preprocess_input_data
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')


def train_and_save_model(df):
    """
    Split the data into train and test sets."""
    df = detect_anomalies_dbscan(df) 
    df = detect_anomalies_isolation_forest(df) 
    X= df[['Claim_Amount', 'Age', 'High_Claim', 'Suspicious_Timing', 'Cost_per_Age', 
            'claim_count', 'avg_gap', 'Medical_History', 'Specialization', 'Reputation_Score','Anomaly_DBSCAN', 'Anomaly_Isolation_Forest']]


    y = df['Fraudulent'] 

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    #Preprocess the data
    X_train_processed, preprocessor =preprocess_input_data(X_train)
    X_test_processed = preprocessor.transform(X_test)


    """
    Train XGBoost and Random Forest models and return a voting classifier."""

    logging.info('Training models...')

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42)

    logging.info('Training xgb models...')
    xgb_model.fit(X_train_processed,y_train)

    #create voting classifier
    # logging.info('Voting Classifier started...')
    # voting_clf = VotingClassifier(estimators=[('rf',rf_model),
    #                                         ('xgb',xgb_model),
    #                                         ('lr', lr_model)],
    #                                         voting='soft' )
    # logging.info('Fit on trained processed data....')
    # voting_clf.fit(X_train_processed,y_train)

    # evaluate the model
    logging.info("Evaluating the model")
    # y_pred = voting_clf.predict(X_test_processed)
    y_pred = xgb_model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Voting Classifier(RF + XGB) Model Accuracy: {accuracy:.2f}")

    logging.info("Saving the model and preprocessor...")
    # joblib.dump(voting_clf,'fraud_detection_model.pkl')
    joblib.dump(xgb_model,'fraud_detection_xgb_model.pkl')
    joblib.dump(preprocessor,'preprocessor.pkl')


    # return voting_clf, preprocessor
    return xgb_model, preprocessor

df= pd.read_csv('data/fraud_insurance_claim.csv')
# Clean and engineer features
df = clean_data(df)
df = feature_engineering(df)
train_and_save_model(df)


