import streamlit as st
import joblib
import numpy as np

# Load trained model
import os
# get path relative to this file
model_path = os.path.join(os.path.dirname(__file__), "ckd_best_model.pkl")
model = joblib.load(model_path)


st.set_page_config(page_title="CKD Prediction App", page_icon="🩺", layout="wide")

st.title("🩺 Chronic Kidney Disease (CKD) Prediction")
st.write("Fill in patient details below and predict whether CKD is present.")

# --- Collect inputs from user ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 45)
    bp = st.number_input("Blood Pressure (mmHg)", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.0, 1.05, step=0.01, format="%.2f")
    al = st.number_input("Albumin", 0, 5, 0)
    su = st.number_input("Sugar", 0, 5, 0)
    rbc = st.selectbox("Red Blood Cells", ("Normal", "Abnormal"))
    pc = st.selectbox("Pus Cell", ("Normal", "Abnormal"))
    pcc = st.selectbox("Pus Cell Clumps", ("Not Present", "Present"))
    ba = st.selectbox("Bacteria", ("Not Present", "Present"))
    bgr = st.number_input("Blood Glucose Random", 50, 500, 100)
    bu = st.number_input("Blood Urea", 1, 400, 50)
    sc = st.number_input("Serum Creatinine", 0, 20, 1)

with col2:
    sod = st.number_input("Sodium", 100, 200, 140)
    pot = st.number_input("Potassium", 2, 10, 4)
    hemo = st.number_input("Hemoglobin", 3, 20, 13)
    pcv = st.number_input("Packed Cell Volume", 20, 55, 40)
    wc = st.number_input("White Blood Cell Count", 2000, 25000, 8000)
    rc = st.number_input("Red Blood Cell Count", 2, 7, 5)
    htn = st.selectbox("Hypertension", ("No", "Yes"))
    dm = st.selectbox("Diabetes Mellitus", ("No", "Yes"))
    cad = st.selectbox("Coronary Artery Disease", ("No", "Yes"))
    appet = st.selectbox("Appetite", ("Good", "Poor"))
    pe = st.selectbox("Pedal Edema", ("No", "Yes"))
    ane = st.selectbox("Anemia", ("No", "Yes"))

# --- Encode categorical inputs ---
def encode_features():
    return [
        age, bp, sg, al, su,
        0 if rbc == "Normal" else 1,
        0 if pc == "Normal" else 1,
        0 if pcc == "Not Present" else 1,
        0 if ba == "Not Present" else 1,
        bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
        1 if htn == "Yes" else 0,
        1 if dm == "Yes" else 0,
        1 if cad == "Yes" else 0,
        1 if appet == "Good" else 0,
        1 if pe == "Yes" else 0,
        1 if ane == "Yes" else 0
    ]

# --- Prediction button ---
if st.button("🔍 Predict CKD"):
    features = np.array(encode_features()).reshape(1, -1)
    
    try:
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("⚠️ The model predicts: **CKD Detected**")
        else:
            st.success("✅ The model predicts: **No CKD**")
    except AttributeError:
        st.error("❌ Model not loaded correctly. Make sure 'ckd_best_model.pkl' is a valid trained ML model.")
