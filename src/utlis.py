import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

#initialize the faker
fake = Faker()

# dataset record and set 10%  fraud probabilty
num_records = 5000
fraud_probabilty = 0.1 

procedure_code_cost_map = {
        '99281': (10000, 20000),
        '99282': (20000, 40000),
        '99283': (40000, 60000),
        '99284': (60000, 80000),
        '99285': (80000, 100000),
    }

Procedure_Codes = {
    'Emergency':["99281", "99282", "99283", "99284", "99285"],  # Emergency visits,
    'Non-Emergency': ["99381", "99382", "99383", "99384", "99385", "11042", "12001", "45378", "93000", "12051", "43235", "10060", "11730", "72148", "71020"]
}
# Generate Synthetic Data 

def generate_synthetic_data(num_records):
    """
        Generate the synthetic data based on the feature information below
    """
    data = []

    for _ in range(num_records):

        #patient claim status
        claim_id = fake.uuid4()[:8]
        procedure_type = random.choice(['Emergency','Non-Emergency'])
        Procedure_Code = random.choice(Procedure_Codes[procedure_type])
        # Claim amount based on procedure and anomalies
        if procedure_type == 'Emergency':
            claim_amount = round(random.uniform(*procedure_code_cost_map[Procedure_Code]), 2)
        else:
            claim_amount = round(random.uniform(1000, 50000), 2)

        # introduce the outlier in claim amount(anomaly)
        if random.random() < 0.02: #2% chance to claim anomaly
            claim_amount = round(random.uniform(200000, 800000), 2)# Extremely hihg claim amount

        claim_status = random.choice(['Approved', "Denied","Under Review"])
        if claim_status == "Denied" and random.random() < 0.3:
            claim_amount = None 
        service_date = fake.date_between(start_date="-5y", end_date = 'today')

        # Patient Information
        patient_id = fake.random_int(min=10000, max=99999)

        # Introduce anomaly in age
        age = random.randint(18,90)
        if random.random() <0.01:
            age= random.choice([1,100])

        # Age impacts Claim Amount
        if claim_amount is not None:
            if age > 50:
                claim_amount += random.uniform(10000, 50000)
            else:
                claim_amount = random.uniform(1000, 15000)
        gender = random.choice(['Male','Female'])
        location= fake.city()
        if age <=5:
            medical_history = 'None'
        else:
            medical_history = random.choice(['Diabetes','Hypertension','Asthama','None'])
        
        
        

        insurance_plan = random.choice(['Basic','Standard', 'Premium'])

        if claim_amount is not None and insurance_plan == "Premium":
            claim_amount = max(claim_amount - 5000, 1000)

        # Provider Information
        provider_id = fake.random_int(min=10000, max=99999)

        specialization = random.choice(["Cardiology", "Orthopedics", "Dermatology", "Emergency Medicine"])
        if specialization == "Emergency Medicine":
            reputation_score = round(random.uniform(3.5, 5.0),2)
        else:
            reputation_score = round(random.uniform(1.0, 5.0), 2)
        provider_location = fake.city()
        # reputation_score = round(random.uniform(1,5),2)

        #Financial indicators
        payment_method = random.choice(["Credit Card", "Debit Card", "Insurance", "Cash"])
        
        
        time_between_claims = random.randint(0,365) # in days
        if random.random() < 0.05:
            time_between_claims = 1 # same day claims to simulate fraud

        # cost_deviation = round(random.uniform(0.8, 1.5),2)
        if procedure_type == "Emergency":
            cost_deviation = round(random.uniform(1.0, 1.5), 2)
        else:
            cost_deviation = round(random.uniform(0.8, 1.2), 2)

        # Fraud Indicators
        fraudulent = 1 if random.random() < fraud_probabilty and time_between_claims < 7 else 0
        suspicious_timing = 1 if time_between_claims < 7 and fraudulent else 0
        duplicate_claims = 1 if fraudulent and random.random() < 0.3 else 0 
        
        if procedure_type == "Emergency":
            cost_deviation = round(random.uniform(1.0, 1.5), 2)
        elif procedure_type == "Non-Emergency" and fraudulent:
            cost_deviation = round(random.uniform(0.5, 1.8), 2)

        if fraudulent:
            cost_deviation = round(random.uniform(0.5, 2.0), 2)
            reputation_score = round(random.uniform(1.0, 3.0), 2)
            time_between_claims = random.randint(1, 7)
        # Ensure that cost_deviation is meaningful only when claim amount is non-zero
        

        # Create a strong correlation between Reputation Score and Cost Deviation
        cost_deviation = reputation_score * 0.3 + random.uniform(0.5, 1.5)  # Correlated with some noise
        

        # Append record

        data.append({
            'Claim ID': claim_id,
            'Claim Amount': claim_amount,
            'Procedure Code': Procedure_Code,
            "Service Date": service_date,
            "Claim Status": claim_status,
            "Patient ID": patient_id,
            "Age": age,
            "Gender": gender,
            "Location": location,
            "Medical History": medical_history,
            "Insurance Plan": insurance_plan,
            "Provider ID": provider_id,
            "Specialization": specialization,
            "Provider Location": provider_location,
            "Reputation Score": reputation_score,
            "Payment Method": payment_method,
            "Time Between Claims": time_between_claims,
            "Cost Deviation": cost_deviation,
            "Suspicious Timing": suspicious_timing,
            "Duplicate Claims": duplicate_claims,
            "Fraudulent": fraudulent
        })

    return pd.DataFrame(data)


def add_missing_values_denied_claims(df):
    """
    Introduces missing values conditionally, such as missing 'Claim Amount' for 'Denied' claims.
    """
    denied_claims = df[df['Claim_Status'] =='Denied']
    missing_index = denied_claims.index

    for idx in missing_index:
        df.at[idx, 'Claim_Amount'] = None
    return df