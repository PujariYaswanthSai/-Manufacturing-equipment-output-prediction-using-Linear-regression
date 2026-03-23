# Project Explanation and CSV Training Note

I confirmed the repository currently has no CSV file, so here is how the project works today and how the same training flow would use a CSV dataset.

## Read metrics.json

Current model metrics from model/metrics.json:
- R2 score: 0.993139
- MAE: 4.655368

## What This Project Predicts

This project predicts manufacturing output (units produced) from three machine conditions:
- Temperature
- Pressure
- Runtime

## Project Layers

It is a full ML app with 3 layers:
- Training script in training/train.py
- Prediction API in backend/main.py
- Web interface in frontend/app.py

## End-to-End Flow

1. The model is trained and saved as files in model/model.pkl and model/scaler.pkl.
2. FastAPI loads those files and serves prediction through /predict in backend/main.py.
3. Streamlit sends user inputs to the API and shows predicted units in frontend/app.py.
4. The app also shows model quality metrics from model/metrics.json.

## How CSV-Based Training Fits In

The current training script uses synthetic data generation. To use a CSV dataset while keeping the same app flow:
1. Read the CSV with pandas.
2. Use Temperature, Pressure, Runtime as input features.
3. Use Units_Produced as the target.
4. Keep the same split, scaling, training, evaluation, and export steps.
5. Save artifacts as model/model.pkl, model/scaler.pkl, and model/metrics.json so backend and frontend continue working without structural changes.
