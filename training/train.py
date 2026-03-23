import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set seed for reproducibility
np.random.seed(42)

# Generate Synthetic Data
n_samples = 1000
temperature = np.random.normal(70, 10, n_samples)
pressure = np.random.normal(50, 5, n_samples)
runtime = np.random.uniform(1, 24, n_samples)

# Output formula: Units = 50 + 2*Temp - 0.5*Pressure + 10*Runtime + noise
units_produced = 50 + 2 * temperature - 0.5 * pressure + 10 * runtime + np.random.normal(0, 10, n_samples)

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
score = model.score(X_test_scaled, y_test)
print(f"Model R^2 Score on Test Set: {score:.4f}")

# Save Model and Scaler
# We traverse up logically to the root directory `pro1`, then `model`
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '..', 'model')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Exported model to: {model_path}")
print(f"Exported scaler to: {scaler_path}")
