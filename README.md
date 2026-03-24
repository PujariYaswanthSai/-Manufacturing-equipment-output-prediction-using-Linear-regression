# Manufacturing Equipment Output Prediction

This project predicts manufacturing output (`Parts_Per_Hour`) from process, machine, material, and time context features.

It includes:
- A training pipeline in `training/train.py`
- A FastAPI inference API in `backend/main.py`
- A Streamlit UI in `frontend/app.py`

## Current Model
- Model: `LinearRegression`
- Preprocessing: `ColumnTransformer`
  - Numeric features scaled with `StandardScaler`
  - Categorical features encoded with `OneHotEncoder(handle_unknown="ignore")`
- Target: `Parts_Per_Hour`
- Data source: `training/manufacturing_dataset_1000_samples.csv`

### Latest Metrics
From `model/metrics.json`:
- `r2_score`: `0.933089`
- `mae`: `2.498103`
- `train_samples`: `408`
- `test_samples`: `102`
- `feature_count`: `21`

## Input Schema Used for Training
CSV columns:
- `Timestamp`
- `Injection_Temperature`
- `Injection_Pressure`
- `Cycle_Time`
- `Cooling_Time`
- `Material_Viscosity`
- `Ambient_Temperature`
- `Machine_Age`
- `Operator_Experience`
- `Maintenance_Hours`
- `Shift`
- `Machine_Type`
- `Material_Grade`
- `Day_of_Week`
- `Temperature_Pressure_Ratio`
- `Total_Cycle_Time`
- `Efficiency_Score`
- `Machine_Utilization`
- `Parts_Per_Hour` (target)

Derived timestamp features used internally:
- `Timestamp_Hour`
- `Timestamp_Day`
- `Timestamp_Month`
- `Timestamp_DayOfWeek`

## Project Structure
```text
this4/
  backend/
    main.py
  frontend/
    app.py
  model/
    model.pkl
    scaler.pkl
    metrics.json
  training/
    manufacturing_dataset_1000_samples.csv
    train.py
  requirements.txt
  render.yaml
  README.md
```

## Local Setup
### 1. Create and activate virtual environment
Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Train model from CSV
```powershell
$env:DATASET_PATH = "./training/manufacturing_dataset_1000_samples.csv"
python training/train.py
```

Artifacts are written to:
- `model/model.pkl`
- `model/scaler.pkl`
- `model/metrics.json`

### 4. Start backend
From project root:
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

API docs:
- `http://127.0.0.1:8000/docs`

### 5. Start frontend
In a new terminal from project root:
```powershell
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

UI URL:
- `http://127.0.0.1:8501`

## API Endpoints
- `GET /` health/welcome message
- `GET /model-metrics` returns training metrics
- `POST /predict` returns:
  - `Units_Produced`

### Prediction Behavior Notes
- Backend normalizes category aliases to training labels (for example `Morning -> Day`, `Type-A -> Type_A`).
- Backend enforces non-negative prediction output by clamping at `0.0`.

## Deployment
`render.yaml` is included for deployment on Render.
