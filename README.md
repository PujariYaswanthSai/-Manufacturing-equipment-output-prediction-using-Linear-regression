# Manufacturing Equipment Output Prediction

This project implements a Machine Learning solution to predict the output (units produced) of manufacturing equipment based on operating parameters. It uses a **Linear Regression** model served by a **FastAPI** backend and provides an interactive user interface built with **Streamlit**.

## Features
- **Machine Learning Model**: Trained on equipment parameters (`Temperature`, `Pressure`, `Runtime`) using `scikit-learn`'s Linear Regression algorithm.
- **RESTful API Backend**: A high-performance Python backend powered by FastAPI that loads the trained `.pkl` models to serve secure predictions natively.
- **Interactive UI**: A Streamlit dashboard where users can input parameters using simple sliders to obtain a real-time prediction output.
- **Ready for Deployment**: Includes a `render.yaml` specification for 1-click cloud deployments on Render.

## Model Training Environment
Model training was performed in **Google Colab** as well as locally. The training logic is available in `training/train.py`, and the generated artifacts are stored in the `model/` directory (`model.pkl` and `scaler.pkl`).

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

## Deployment
This project is configured for cloud deployment. You can easily host it on [Render](https://render.com) by linking this repository. Render will automatically detect the `render.yaml` file, spin up a web service for the FastAPI backend, and another web service for the Streamlit UI.
