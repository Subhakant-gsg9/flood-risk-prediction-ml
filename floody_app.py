import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------
#  Train the Flood Prediction Model
# ---------------------------
@st.cache_data
def load_and_train_model():
    df = pd.read_csv(r"C:\Users\subha\OneDrive\Desktop\ML_Project\flood\reduced_flood.csv")
    target = 'FloodProbability'
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, list(X.columns)

model, feature_names = load_and_train_model()

# ---------------------------
#  Streamlit App Interface
# ---------------------------
st.title("🌧️ Flood Probability Prediction App")
st.markdown("Enter the values for each environmental and infrastructure feature:")

user_input = []
for feature in feature_names:
    value = st.slider(f"{feature}", min_value=0, max_value=10, value=5)
    user_input.append(value)

# ---------------------------
#  User Type Dropdown
# ---------------------------
user_type = st.selectbox("👤 Select Your Profile", ["General", "Farmer", "Family", "Elderly"])

# ---------------------------
#  Predict Button
# ---------------------------
# ---------------------------
# Session State Setup
# ---------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "show_measures" not in st.session_state:
    st.session_state.show_measures = False

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🚨 Predict Flood Probability"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    st.session_state.prediction = model.predict(input_df)[0]
    st.session_state.show_measures = False  # Reset safety measure view

# Show prediction if available
if st.session_state.prediction is not None:
    prediction = st.session_state.prediction
    st.success(f"🔎 Predicted Flood Probability: **{prediction:.4f}**")

    if prediction < 0.3:
        st.info("☔ Low Risk")
    elif prediction < 0.6:
        st.warning("⛈️ Medium Risk")
    else:
        st.error("🌊 High Risk")

# ---------------------------
# Show Safety Measures Button
# ---------------------------
if st.session_state.prediction is not None and st.button("🛡️ Show Safety Measures"):
    st.session_state.show_measures = True

if st.session_state.show_measures and st.session_state.prediction is not None:
    prediction = st.session_state.prediction
    risk_level = ""
    if prediction < 0.3:
        risk_level = "Low"
    elif prediction < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    st.subheader(f"📋 Safety Advice for {user_type} - {risk_level} Risk")

    advice = {
        "General": {
            "Low": "✅ Stay informed.\n✅ Keep emergency contacts handy.\n✅ Watch local weather alerts.",
            "Medium": "⚠️ Prepare emergency kit.\n⚠️ Stay ready to evacuate.\n⚠️ Avoid waterlogged areas.",
            "High": "🚨 Evacuate immediately.\n🚨 Turn off gas/electricity.\n🚨 Follow government instructions.",
        },
        "Farmer": {
            "Low": "✅ Secure storage for seeds/tools.\n✅ Monitor irrigation levels.",
            "Medium": "⚠️ Move livestock to safe zones.\n⚠️ Cover crop supplies.\n⚠️ Inspect water channels.",
            "High": "🚨 Evacuate livestock.\n🚨 Protect tractors/equipment.\n🚨 Stay in touch with agri dept.",
        },
        "Family": {
            "Low": "✅ Keep basic supplies.\n✅ Keep family informed.\n✅ Prepare child safety plan.",
            "Medium": "⚠️ Pack essentials.\n⚠️ Stay in touch with relatives.\n⚠️ Charge phones and power banks.",
            "High": "🚨 Evacuate to a safe shelter.\n🚨 Carry documents and meds.\n🚨 Avoid floodwaters.",
        },
        "Elderly": {
            "Low": "✅ Keep medications close.\n✅ Stay in touch with a neighbor.",
            "Medium": "⚠️ Have help on standby.\n⚠️ Pack health documents.\n⚠️ Limit movement outdoors.",
            "High": "🚨 Call for immediate help.\n🚨 Avoid stairs/water.\n🚨 Stay with a trusted caregiver.",
        }
    }

    st.markdown(advice[user_type][risk_level])
                
