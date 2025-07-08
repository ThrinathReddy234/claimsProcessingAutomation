import streamlit as st
import pandas as pd
import joblib
from modules.document_processing import load_and_validate_data
from modules.feature_extraction import preprocess_data
from modules.workflow_routing import score_complexity
from modules.visualizations import generate_visualizations

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("📋 Claims Automation Dashboard")

uploaded_file = st.file_uploader("Upload Claims CSV", type="csv")

if uploaded_file:
    df_raw = load_and_validate_data(uploaded_file)
    st.write("### Raw Uploaded Data", df_raw.head())

    df_clean = preprocess_data(df_raw.copy())
    X = df_clean.drop('fraud_reported', axis=1, errors='ignore')
    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    df_raw['predicted_fraud_prob'] = y_prob
    df_raw['claim_complexity_score'] = [
        score_complexity(row, prob) for row, prob in zip(df_raw.to_dict(orient='records'), y_prob)
    ]
    df_raw['claim_routing_action'] = df_raw['claim_complexity_score'].apply(
        lambda x: 'HUMAN_REVIEW' if x > 40 else 'AUTO_PROCESS'
    )

    st.success("Routing Completed...")
    st.write(df_raw[['claim_complexity_score', 'claim_routing_action']].head())

    generate_visualizations(df_raw)

    st.image("output/complexity_distribution.png")
    st.image("output/routing_decision_pie.png")
    st.image("output/fraud_probability_distribution.png")