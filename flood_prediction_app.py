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
st.title(" Flood Probability Prediction App")
st.markdown("Enter the values for each environmental and infrastructure feature:")

user_input = []
for feature in feature_names:
    value = st.slider(f"{feature}", min_value=0, max_value=10, value=5)
    user_input.append(value)

# ---------------------------
#  Predict Button
# ---------------------------
if st.button("Predict Flood Probability"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    prediction = model.predict(input_df)[0]
    
    st.success(f" Predicted Flood Probability: **{prediction:.4f}**")

    if prediction < 0.3:
        st.info(" ☔Low Risk")
    elif prediction < 0.6:
        st.warning(" ⛈️Medium Risk")
    else:
        st.error(" 🌊High Risk")
