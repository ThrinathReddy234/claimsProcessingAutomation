from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from modules.feature_extraction import preprocess_data
from modules.workflow_routing import score_complexity
from fastapi.responses import JSONResponse

app = FastAPI()
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/")
def root():
    return {"message": "Claims Routing API is running..."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df_clean = preprocess_data(df)
        X = df_clean.drop("fraud_reported", axis=1, errors="ignore")
        X_scaled = scaler.transform(X)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        df["predicted_fraud_prob"] = y_prob
        df["claim_complexity_score"] = [
            score_complexity(row, prob) for row, prob in zip(df.to_dict(orient="records"), y_prob)
        ]
        df["claim_routing_action"] = df["claim_complexity_score"].apply(
            lambda x: "HUMAN_REVIEW" if x > 40 else "AUTO_PROCESS"
        )

        result = df[["claim_complexity_score", "claim_routing_action"]].to_dict(orient="records")
        return JSONResponse(content={"results": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)