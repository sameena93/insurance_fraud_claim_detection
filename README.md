# Fraud Detection for Healthcare Insurance Claim

![Click to Explore End to End Project](https://insurancefraudclaimdetection.streamlit.app/)(https://insurancefraudclaimdetection.streamlit.app/)

### *Problem:* 
* Healthcare insurance fraud leads to billions in losses annually. It aims to reduce losses due to fraud while ensuring legitimate claims are processed efficiently
* *Fraud Types*:
    * Billing for services not provided.
    * Inflated bills or duplicate claims
    * Flasified patient records
    * Unnecessary procedure or prescriptions.
      
## Dataset 
* Used Synthetic Data Used for this Problem

###  Methodology
* Data Cleaning & Exploration
* Feature Engineering using statistic technique
* Unspuervised ML- DBSCAN, Isolation Forest
* Model Building (e.g., XGB, Random Forest, Logistic Regression)
* Evaluation (F1 score, precision, recall, Type1 error, TypeII error, ROC AUC Curve)
* Dockerization, CI/CD Pipeline, Git , GitHub , Hugging Face Deployment




---
# Exploratory Data Analysis (EDA)

## 1. Claim Amount Distribution by Claim Status
![Claim Amount Distribution by Claim Status](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/claim_amount_status.png)

---

## 2. Claim Amount Distribution(KDE)
![Kde](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/CLaim_amount_distibution(KDE).png)

---

## 3.  Response Variable Class
#### Shows Imbalance class- Used SMOTE (Synthetic Minority Oversampling Technique)  technique to handle the imbalance class.
![fraud_imbalnce](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/Fraudulent_class_imbalance.png)

---

## 4. Correlation Matrix
![corr](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/Correlation_matrix.png)

--- 

### 5. Payment Method Distibution
![](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/Payment_method_distribution.png)

---

### 6. Claim Status Distribution 
![](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/claim_status_distribution.png)

---

# Precision Recall and ROC AUC Curve 
#### Logistic Regression Model

![lr](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/LR_EG.png)

#### Random Forest Model
![rf](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/random%20forest_eg.png)

#### XGB Model
![xgb](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/xgb_eg.png)


---


# Evaluation Matrix 
## Before SMOTE
![before smote](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/Before_SMOTE.png)

## After SMOTE
![after smote](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/after_SMOTE.png)

---
# Conclusion
### Impact of SMOTE on Model Performance
Applying SMOTE (Synthetic Minority Oversampling Technique) significantly enhanced the performance of all models in terms of recall, which is crucial for fraud detection as it ensures fewer fraudulent claims are missed. 

# Key Observations
1. ***XGB Model:***

Performed the best after SMOTE with a balanced F1 score of 0.78.
Improved recall (0.77) without a significant drop in precision (0.80), making it the most suitable model for deployment.

2. ***Random Forest Model:***

Achieved the highest recall (0.85) after SMOTE but had a notable drop in precision (0.58).
The F1 score improved from 0.62 to 0.69, indicating better overall performance post-SMOTE.

3. ***Logistic Regression Model:***

Recall increased significantly (from 0.35 to 0.96), but the drop in precision (from 0.69 to 0.49) resulted in more false positives.
Despite this, the F1 score improved from 0.46 to 0.65, showing a net positive impact.


---
-- ![Watch end to end project video](https://github.com/sameena93/insurance_fraud_claim_detection/blob/main/static/video.mp4)
