from fastapi import FastAPI
import pandas as pd

from src.serving.model.load_model import model
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.serving.inference import preprocess_input

THRESHOLD = 0.35

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def home():
    return{
        "message":"Churn Prediction API running"
    }

@app.post("/predict")
def predict(data:dict):
    df = pd.DataFrame([data])
    df = preprocess_input(df)
    print(df.dtypes)
    print(df.columns.tolist())
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability>=THRESHOLD)
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }