import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import json
import os

# Set seed for reproducibility
np.random.seed(42)

# Generate Synthetic Data
n_samples = 3000
temperature = np.random.normal(70, 10, n_samples)
pressure = np.random.normal(50, 5, n_samples)
runtime = np.random.uniform(1, 24, n_samples)

# Output formula: Units = 50 + 2*Temp - 0.5*Pressure + 10*Runtime + noise
units_produced = 50 + 2 * temperature - 0.5 * pressure + 10 * runtime + np.random.normal(0, 6, n_samples)

df = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Runtime': runtime,
    'Units_Produced': units_produced
})

print("Generated synthetic dataset. First 5 rows:")
print(df.head())

# Features and Target
X = df[['Temperature', 'Pressure', 'Runtime']]
y = df['Units_Produced']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model R^2 Score on Test Set: {r2:.4f}")
print(f"Model MAE on Test Set: {mae:.4f}")

# Save Model and Scaler
# We traverse up logically to the root directory `pro1`, then `model`
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '..', 'model')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
metrics_path = os.path.join(model_dir, 'metrics.json')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

metrics = {
    "model_name": "LinearRegression",
    "r2_score": round(float(r2), 6),
    "mae": round(float(mae), 6),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "trained_with": "scikit-learn",
    "trained_in": "Google Colab and local Python environments"
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

print(f"Exported model to: {model_path}")
print(f"Exported scaler to: {scaler_path}")
print(f"Exported metrics to: {metrics_path}")
