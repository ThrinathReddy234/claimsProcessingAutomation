import pandas as pd
import numpy as np

def preprocess_data(df):
    drop_cols = ['_c39', 'policy_number', 'insured_zip', 'incident_location']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')

    df['months_bw_bind_incident'] = (
        (df['incident_date'].dt.year - df['policy_bind_date'].dt.year) * 12 +
        (df['incident_date'].dt.month - df['policy_bind_date'].dt.month)
    ).fillna(0).astype(int)

    cat_cols = ['collision_type', 'property_damage', 'police_report_available']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].replace('?', np.nan)
            df[col] = df[col].fillna(df[col].mode()[0])

    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
    df.dropna(thresh=len(df.columns) - 5, inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    df.drop(['policy_bind_date', 'incident_date'], axis=1, inplace=True, errors='ignore')

    return df
