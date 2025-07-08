from modules.document_processing import load_and_validate_data
from modules.feature_extraction import preprocess_data
from modules.decision_engine import train_models
from modules.workflow_routing import score_complexity
from modules.visualizations import generate_visualizations

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_pipeline(csv_path):
    df = load_and_validate_data(csv_path)
    df = preprocess_data(df)

    X = df.drop('fraud_reported', axis=1)
    y = df['fraud_reported']

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_models(X_train_scaled, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    X_test_reset = X_test.reset_index(drop=True)
    X_test_reset['predicted_fraud_prob'] = y_prob
    X_test_reset['claim_complexity_score'] = [
        score_complexity(row, prob) for row, prob in zip(X_test_reset.to_dict(orient='records'), y_prob)
    ]
    X_test_reset['claim_routing_action'] = X_test_reset['claim_complexity_score'].apply(
        lambda x: 'HUMAN_REVIEW' if x > 40 else 'AUTO_PROCESS'
    )

    output_df = X_test_reset[['claim_complexity_score', 'claim_routing_action']]
    os.makedirs('output', exist_ok=True)
    output_df.to_csv('output/output_routing_decisions.csv', index=False)

    generate_visualizations(X_test_reset)

if __name__ == "__main__":
    run_pipeline('dataset/insurance_claims.csv')