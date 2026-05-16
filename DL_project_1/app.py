import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

@st.cache_resource(show_spinner="Loading Model and Scaler...")
def load_app_resources():
    # Load Keras Model
    model = load_model("breast_cancer_model.keras")
    
    # Load Scaler
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
        
    return model, scaler

try:
    model, scaler = load_app_resources()
except Exception as e:
    st.error(f"Error loading files: {e}. Please ensure 'breast_cancer_model.keras' and 'scaler.pkl' exist.")
    st.stop()

# Load dataset metadata for feature names and target labels
data = load_breast_cancer()
feature_names = data.feature_names
target_names = data.target_names

st.title("🩺 Breast Cancer Classification (Neural Network)")
st.markdown("Enter the tumor features in the sidebar to predict if it is **Malignant** or **Benign**.")

# Sidebar for inputs
st.sidebar.header("Input Features")
st.sidebar.write("Adjust the values below (defaults are dataset means):")

input_data = {}
# Grouping inputs to keep UI concise
categories = ["mean", "error", "worst"]

for cat in categories:
    with st.sidebar.expander(f"{cat.capitalize()} Features", expanded=(cat=="mean")):
        for feature in feature_names:
            if cat in feature:
                # Default value from the dataset mean
                idx = list(feature_names).index(feature)
                default_val = float(np.mean(data.data[:, idx]))
                input_data[feature] = st.number_input(feature, value=default_val, key=feature)

# Predict Button
if st.button("🔍 Predict Tumor Type", use_container_width=True):
    # Prepare data for prediction
    input_df = pd.DataFrame([input_data])[feature_names]
    scaled_input = scaler.transform(input_df)
    
    # Keras prediction
    prediction_probs = model.predict(scaled_input)
    prediction_label = np.argmax(prediction_probs)
    prob = prediction_probs[0][prediction_label]
    
    result = target_names[prediction_label].capitalize()
    
    st.markdown("### Prediction Result")
    if prediction_label == 0: # Malignant
        st.error(f"🚨 **{result}** (Confidence: {prob:.2%})")
        st.warning("Please consult a healthcare professional for further evaluation.")
    else: # Benign
        st.success(f"✅ **{result}** (Confidence: {prob:.2%})")
        st.info("The mass is predicted to be benign. Regular check-ups are still recommended.")

st.markdown("---")
st.write("### Dataset Preview")
st.dataframe(pd.DataFrame(data.data, columns=feature_names).head())
