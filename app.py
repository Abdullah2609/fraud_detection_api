from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Load trained model
model = joblib.load("model/fraud_model.pkl")

class FraudInput(BaseModel):
    TX_HOUR: float
    TX_DOW: float
    CUSTOMER_TX_COUNT_1D: int
    CUSTOMER_TX_COUNT_7D: int
    CUSTOMER_TX_COUNT_30D: int
    TERMINAL_FRAUD_COUNT_1D: int
    TERMINAL_FRAUD_COUNT_7D: int
    TERMINAL_FRAUD_COUNT_30D: int

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: FraudInput):
    # Convert input into DataFrame
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    
    prob = model.predict_proba(df)[0][1]  # probability of fraud (class=1)

    return {
        "prediction": int(prediction),
        "fraud_probability": float(prob)
    }
