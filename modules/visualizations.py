import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(df):
    os.makedirs('output', exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df['claim_complexity_score'], bins=10, kde=True, color='skyblue')
    plt.title('Claim Complexity Score Distribution')
    plt.savefig('output/complexity_distribution.png')
    plt.close()

    plt.figure(figsize=(6, 6))
    df['claim_routing_action'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'orange'])
    plt.title('Claim Routing Decision Breakdown')
    plt.ylabel('')
    plt.savefig('output/routing_decision_pie.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df['predicted_fraud_prob'], bins=20, color='red', kde=True)
    plt.title('Predicted Fraud Probability Distribution')
    plt.savefig('output/fraud_probability_distribution.png')
    plt.close()