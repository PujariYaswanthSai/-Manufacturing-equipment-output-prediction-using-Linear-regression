import streamlit as st
import requests

# Define the backend API URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Manufacturing Output Predictor", layout="centered")

st.title("🏭 Manufacturing Equipment Output Prediction")
st.write("Predict the number of units produced based on equipment operating parameters.")

st.sidebar.header("Equipment Parameters")

# Input fields
temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0,
    help="Operating temperature of the equipment."
)

pressure = st.sidebar.number_input(
    "Pressure (psi)",
    min_value=0.0,
    max_value=150.0,
    value=50.0,
    step=1.0,
    help="Operating pressure."
)

runtime = st.sidebar.number_input(
    "Runtime (hours)",
    min_value=0.0,
    max_value=24.0,
    value=12.0,
    step=0.5,
    help="Continuous runtime of the equipment."
)

st.write("### Current Input Parameters")
st.write(f"- **Temperature**: {temperature} °C")
st.write(f"- **Pressure**: {pressure} psi")
st.write(f"- **Runtime**: {runtime} hours")

if st.button("Predict Output", type="primary"):
    # Prepare the payload
    payload = {
        "Temperature": temperature,
        "Pressure": pressure,
        "Runtime": runtime
    }
    
    with st.spinner("Calculating prediction..."):
        try:
            # Make the request to the FastAPI backend
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                units_produced = result.get("Units_Produced", 0.0)
                
                st.success("Prediction retrieved successfully!")
                st.metric(label="Predicted Units Produced", value=f"{units_produced:.2f}")
                
            else:
                st.error(f"Error from server: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend server. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")
