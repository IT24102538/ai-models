# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load model
MODEL_PATH = "house_cost_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(title="Sri Lanka House Cost Prediction API")

# Request schema
class HouseRequest(BaseModel):
    bedrooms: int
    bathrooms: int
    floors: int
    area: int
    material: str
    cement_bags: int
    steel_kg: int
    bricks: int
    workers: int
    days: int

# Prediction endpoint
@app.post("/predict")
def predict(req: HouseRequest):
    try:
        # Convert input to dataframe
        df = pd.DataFrame([{
            "Bedrooms": req.bedrooms,
            "Bathrooms": req.bathrooms,
            "Floors": req.floors,
            "Area_sqft": req.area,
            "CementBags": req.cement_bags,
            "SteelKg": req.steel_kg,
            "BrickCount": req.bricks,
            "Workers": req.workers,
            "LaborDays_est": req.days,
            "LaborCost_LKR": req.workers * req.days * 1200,  # example labor cost
            "MaterialQuality_Low": 1 if req.material=="Low" else 0,
            "MaterialQuality_Medium": 1 if req.material=="Medium" else 0,
            "MaterialQuality_High": 1 if req.material=="High" else 0
        }])
        
        # Make sure all model features are present
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]
        
        # Predict
        cost = model.predict(df)[0]
        return {"estimated_cost_LKR": round(cost)}
    
    except Exception as e:
        return {"error": str(e)}
