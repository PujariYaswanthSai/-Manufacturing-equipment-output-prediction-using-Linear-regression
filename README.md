# Manufacturing Equipment Output Prediction

This project implements a Machine Learning solution to predict the output (units produced) of manufacturing equipment based on operating parameters. It uses a **Linear Regression** model served by a **FastAPI** backend and provides an interactive user interface built with **Streamlit**.

## Features
- **Machine Learning Model**: Trained on equipment parameters (`Temperature`, `Pressure`, `Runtime`) using `scikit-learn`'s Linear Regression algorithm.
- **RESTful API Backend**: A high-performance Python backend powered by FastAPI that loads the trained `.pkl` models to serve secure predictions natively.
- **Interactive UI**: A Streamlit dashboard where users can input parameters using simple sliders to obtain a real-time prediction output.
- **Ready for Deployment**: Includes a `render.yaml` specification for 1-click cloud deployments on Render.

## Model Training Environment
Model training was performed in **Google Colab** as well as locally. The training logic is available in `training/train.py`, and the generated artifacts are stored in the `model/` directory (`model.pkl` and `scaler.pkl`).

## What This Project Does
This project estimates manufacturing equipment output (units produced) from three operating parameters:
- `Temperature`
- `Pressure`
- `Runtime`

It has three connected parts:
- **Training Layer** (`training/train.py`): Trains a regression model and exports artifacts.
- **API Layer** (`backend/main.py`): Loads model artifacts and serves predictions through FastAPI endpoints.
- **Web App Layer** (`frontend/app.py`): Lets users enter machine parameters and view predicted output and model metrics.

Typical runtime flow:
1. Train and export model files into the `model/` folder.
2. FastAPI loads the model and scaler on startup.
3. Streamlit sends user input to `/predict` and shows the predicted units.
4. Streamlit fetches `/model-metrics` to display test metrics such as R2 and MAE.

## Project Structure
```text
pro1/
│
├── backend/
│   └── main.py              # FastAPI application and prediction endpoint
├── frontend/
│   └── app.py               # Streamlit interactive dashboard
├── model/
│   ├── model.pkl            # Serialized Linear Regression Model
│   └── scaler.pkl           # Serialized StandardScaler
├── training/
│   └── train.py             # Script to generate synthetic data and train the model
│
├── requirements.txt         # Project dependencies
├── render.yaml              # Render.com deployment configuration
├── .gitignore               # Git ignored files
└── README.md                # Project documentation
```

## Running the Project Locally

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Manufacturing-equipment-output-prediction-using-Linear-regression
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv .venv
```

On Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Backend
From the project root:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Backend docs will be available at `http://127.0.0.1:8000/docs`.

### 5. Run the Streamlit Frontend
Open a **new terminal** in the same project root and run:
```bash
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```
Then open `http://127.0.0.1:8501` in your browser.

## Training the Model
If you'd like to retrain the model or alter the synthetic data generation logic, simply run:
```bash
python training/train.py
```
This automatically updates the `.pkl` files inside the `model/` folder. Be sure to restart the FastAPI backend if you change the model.

If training in Google Colab, export the updated `model.pkl` and `scaler.pkl` files and replace the files in the local `model/` directory before starting the backend.

## How Training Works with CSV Data
Current state:
- The existing `training/train.py` script generates **synthetic data** using NumPy.
- There is currently no CSV dataset committed in this repository.

How CSV-based training would work with the same pipeline:
1. Load CSV using pandas (for example: `df = pd.read_csv("your_file.csv")`).
2. Set feature columns as `Temperature`, `Pressure`, and `Runtime`.
3. Set target column as `Units_Produced`.
4. Perform train-test split (`train_test_split`).
5. Scale features using `StandardScaler`.
6. Train a `LinearRegression` model.
7. Evaluate using R2 and MAE.
8. Save `model.pkl`, `scaler.pkl`, and `metrics.json` in the `model/` folder.

This means the backend and frontend code can remain unchanged as long as the exported artifacts keep the same names and schema.

## Deployment
This project is configured for cloud deployment. You can easily host it on [Render](https://render.com) by linking this repository. Render will automatically detect the `render.yaml` file, spin up a web service for the FastAPI backend, and another web service for the Streamlit UI.
