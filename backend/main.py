from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import os

app = FastAPI(title="Manufacturing Output Prediction API")

# Define the input data schema
class PredictionInput(BaseModel):
    Temperature: float
    Pressure: float
    Runtime: float

# Define the output data schema
class PredictionOutput(BaseModel):
    Units_Produced: float

# Load the model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '..', 'model')

model_path = os.path.join(model_dir, 'model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
metrics_path = os.path.join(model_dir, 'metrics.json')

model = None
scaler = None
model_metrics = {}

@app.on_event("startup")
def load_artifacts():
    global model, scaler, model_metrics
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                model_metrics = json.load(f)
        else:
            model_metrics = {}
        print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model/scaler artifacts from {model_dir}. Exception: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Manufacturing Equipment Output Prediction API. Use the /predict endpoint to get outputs."}

@app.get("/model-metrics")
def get_model_metrics():
    if not model_metrics:
        return {
            "message": "Model metrics are not available. Run training/train.py to generate metrics.",
            "available": False
        }
    return {"available": True, **model_metrics}

@app.post("/predict", response_model=PredictionOutput)
def predict_output(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
    
    # Prepare data for prediction
    features = [[input_data.Temperature, input_data.Pressure, input_data.Runtime]]
    
    try:
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        
        return PredictionOutput(Units_Produced=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# To run the app use: uvicorn backend.main:app --reload
