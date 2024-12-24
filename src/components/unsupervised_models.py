import logging
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
# import keras
# import tensorflow as tf
# from tensorflow.keras.models import Model


# Setup logging
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')



def detect_anomalies_dbscan(df):
    logging.info("Starting DBSCAN anomaly detection...")
    dbscan = DBSCAN(eps=0.5, min_samples = 5)
    df['Anomaly_DBSCAN'] = dbscan.fit_predict(df[['Claim_Amount', 'Time_Between_Claims']])
    df['Anomaly_DBSCAN'] = df['Anomaly_DBSCAN'].map({-1:1, 1:0})
    logging.info('DBSCAN anomaly detection completed.')
    return df

def detect_anomalies_isolation_forest(df):
    logging.info("Starting Isolation Forest Anomlay Detection...")
    isolation_forest = IsolationForest(contamination=0.05) # 5% fraud detection
    df['Anomaly_Isolation_Forest'] = isolation_forest.fit_predict(df[['Claim_Amount', 'Time_Between_Claims']])
    df['Anomaly_Isolation_Forest'] = df['Anomaly_Isolation_Forest'].map({-1: 1, 1: 0})
    logging.info('Isolation Forest anomaly detection completed.')
    return df


