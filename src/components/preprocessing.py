from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


def preprocess_input_data(df):
    """
    Feature encoding and scaling for the input data."""

    # define categorical and numerical columns
    categorical_columns = ['High_Claim','Suspicious_Timing', 'Specialization']

    numerical_columns = ['Claim_Amount','Age','Cost_per_Age','claim_count','avg_gap','Reputation_Score']

   
    #Create column transformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num',StandardScaler(), numerical_columns),
            ('cat',OneHotEncoder(sparse_output=False), categorical_columns)
        ]
    )

    processed_data = preprocessor.fit_transform(df)

    return processed_data, preprocessor

