
# Claims Processing Automation and Prioritization

This project automates the processing of insurance claims using machine learning. It decides whether a claim can be auto-processed or should be sent for human review based on a calculated complexity score.

## How the System Works

1. **Data Loading**: Loads the claims dataset from CSV format.
2. **Data Preprocessing**: Cleans and encodes the data, handles missing values, date formats, and categorical variables.
3. **Model Training**: Trains multiple models and selects the best one using AUC score this used 70% data for training from the given dataset.
4. **Prediction**: Calculates fraud probability from 30% of the data using the trained model.
5. **Complexity Scoring**: Applies a rule-based scoring mechanism on each claim based on multiple factors.
6. **Routing Decision**: Assigns a label: either 'HUMAN_REVIEW' or 'AUTO_PROCESS' based on complexity score.
7. **Output Generation**: Saves decision results and plots to the output folder.

## How to Run

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Main Pipeline

```bash
python main.py
```

This will process the dataset in `data/insurance_claims.csv`, train models, route claims, and generate visualizations in the `output/` directory.

### 3. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

- Upload a CSV file containing claims
- View predictions, routing decisions, and visualizations

### 4. Run the FastAPI Server

```bash
uvicorn api.server:app --reload
```

Send POST requests to `/predict/` with a CSV file to receive routing decisions via API.

## Routing Logic

The system uses a combination of predicted fraud probability and business logic to determine complexity. Scoring includes:

- +50 if fraud probability > 0.5
- +20 if policy deductible > 1000
- +10 if premium > 1000
- +10 if more than one vehicle involved
- +10 if witnesses are present

If score > 40 → Human Review, else → Auto Process

## Output Files

- `output/output_routing_decisions.csv` - Contains complexity scores and final decisions
- Visualizations:
  - `complexity_distribution.png`
  - `fraud_probability_distribution.png`
  - `routing_decision_pie.png`

## Dataset

The dataset contains fields like:

- Customer details: `age`, `months_as_customer`
- Policy: `policy_deductable`, `policy_annual_premium`
- Incident: `incident_type`, `collision_type`, `property_damage`
- Vehicle: `auto_make`, `auto_model`, `auto_year`
- Target: `fraud_reported`

## Notes

- All input data should follow the same structure as the training data.
- The model expects columns in the same format after preprocessing.