import pandas as pd

def load_and_validate_data(file_path):
    try:
        df = pd.read_csv(file_path)
        assert not df.empty, "Dataset is empty"
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")

    drop_cols = ['_c39', 'policy_number', 'insured_zip', 'incident_location']
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df