import logging
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from src.components.data_ingestion import feature_engineering, clean_data
from src.components.unsupervised_models import detect_anomalies_dbscan, detect_anomalies_isolation_forest
from src.components.preprocessing import preprocess_input_data
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters for tuning
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

param_grid_lr = {
    'C': [0.1, 1.0, 10],
    'class_weight': ['balanced', None],  # Add class_weight to address imbalance
}
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
    X_train_processed, preprocessor = preprocess_input_data(X_train)
    pickle.dump(preprocessor, open("preprocessor.pkl", "wb"))

    X_test_processed = preprocessor.transform(X_test)


    #Apply SMOTE
    smote = SMOTE(random_state = 42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    print(f"Original training set size: {X_train.shape[0]} samples")
    print(f"Resampled training set size: {X_train_resampled.shape[0]} samples")


    """
    Train XGBoost and Random Forest models and return a voting classifier."""

    logging.info('Training models...')

    rf_model = RandomForestClassifier(n_estimators=200,max_depth = 5, random_state=42)
    xgb_model = XGBClassifier(n_estimators=500,max_depth = 7, learning_rate= 0.01,random_state=42,objective='binary:logistic')
    lr_model = LogisticRegression(random_state=42)

    logging.info('Training xgb models...')
    xgb_model.fit(X_train_resampled, y_train_resampled )
    rf_model.fit(X_train_resampled, y_train_resampled )
    lr_model.fit(X_train_resampled, y_train_resampled )
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


    for model, name in zip([xgb_model, rf_model, lr_model], ["XGB Model", "Random Forest Model", "Logistic Regression Model"]):
        logging.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test_processed)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')  # Use 'binary' for binary classification
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
        
        # Print metrics
        print(f"\n{name} Evaluation Metrics:")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Plot ROC Curve  ## FPR (False Positive Rate) and TPR (True Positive Rate) 
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_processed)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend(loc='lower right')
        # plt.show()

        # Save plot 
        plt.savefig(f'ROC_Curve_{name}.png')
        plt.close()

        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, model.predict_proba(X_test_processed)[:, 1])
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f'{name} (PR AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {name}')
        plt.legend(loc='lower left')
        # plt.show()

        # save figure
        plt.savefig(f'Precision_Recall_Curve_{name}.png')
        plt.close()

    logging.info("Saving the model and preprocessor...")
    # joblib.dump(voting_clf,'fraud_detection_model.pkl')
    joblib.dump(xgb_model, 'fraud_detection_xgb_model.pkl')
    joblib.dump(rf_model, 'fraud_detection_rf_model.pkl')
    joblib.dump(lr_model, 'fraud_detection_lr_model.pkl')
    


    # return voting_clf, preprocessor
    return xgb_model, rf_model, lr_model, preprocessor

df= pd.read_csv('data/fraud_insurance_claim.csv')
# Clean and engineer features
df = clean_data(df)
df = feature_engineering(df)

train_and_save_model(df)


# python src/components/with_smote_technique.py