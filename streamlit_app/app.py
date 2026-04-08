import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
import tensorflow as tf
import shap

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Diabetes AI", layout="wide")

# ===============================
# LIGHT THEME CSS
# ===============================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1, h2, h3, h4 {
    color: #1f2937;
}
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.big-number {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# PATHS
# ===============================
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        ARTIFACTS_DIR / "dnn_model_fixed.keras",
        compile=False
    )

    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    with open(ARTIFACTS_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    return model, scaler, meta

model, scaler, meta = load_artifacts()

# ===============================
# SHAP
# ===============================
@st.cache_resource
def load_shap_explainer(model, n_features):
    background = np.zeros((50, n_features))
    return shap.KernelExplainer(lambda x: model.predict(x), background)

shap_explainer = load_shap_explainer(model, len(meta["feature_names"]))

# ===============================
# INPUT PREP
# ===============================
def prepare_input(user_input):
    df = pd.DataFrame([user_input])

    for col in meta["original_columns"]:
        if col not in df.columns:
            df[col] = 0

    df = df[meta["original_columns"]]
    df = pd.get_dummies(df)
    df = df.reindex(columns=meta["feature_names"], fill_value=0)

    return df

# ===============================
# REPORT FUNCTION
# ===============================
def generate_report(input_data, prob, risk_level, top_features):
    return f"""
Diabetes Prediction Report

Age: {input_data['age']}
BMI: {input_data['bmi']}
HbA1c: {input_data['HbA1c_level']}
Glucose: {input_data['blood_glucose_level']}

Risk Level: {risk_level}
Probability: {prob:.2f}

Top Factors: {', '.join(top_features)}

NOTE: This is not a medical diagnosis.
"""

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align:center;'>🧬 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
hba1c = st.sidebar.number_input("HbA1c", 3.0, 15.0, 5.5)
glucose = st.sidebar.number_input("Glucose", 50, 300, 120)

hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoking = st.sidebar.selectbox("Smoking", ["never", "former", "current"])

predict_btn = st.sidebar.button("🚀 Predict")

# ===============================
# LAYOUT
# ===============================


# ===============================
# MAIN LOGIC
# ===============================
if predict_btn:

    input_data = {
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "gender": gender,
        "smoking_history": smoking
    }

    df_input = prepare_input(input_data)

    X = df_input[meta["feature_names"]]
    X_scaled = scaler.transform(X)

    prob = float(model.predict(X_scaled)[0][0])
    pred = int(prob >= 0.5)

    # ----------------------
    # Risk logic
    # ----------------------
    if prob < 0.2:
        risk_level = "🟢 Low Risk"
    elif prob < 0.5:
        risk_level = "🟡 Moderate Risk"
    else:
        risk_level = "🔴 High Risk"

    # ----------------------
    # SHAP
    # ----------------------
    shap_values = shap_explainer.shap_values(X_scaled)
    shap_vals = shap_values[0].flatten()

    shap_df = pd.DataFrame({
        "Feature": meta["feature_names"],
        "Importance": np.abs(shap_vals)
    }).sort_values(by="Importance", ascending=False)

    top_features = shap_df.head(3)["Feature"].tolist()

    # ----------------------
    # METRICS PANEL (RIGHT SIDE)
    # ----------------------
   

    st.markdown("### 📈 Key Insights")

        # Risk + Probability
    st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span><b>Risk Level</b></span>
            <span style="color:{'#22c55e' if 'Low' in risk_level else '#ef4444'};">
                {risk_level}
            </span>
        </div>

        <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
            <span><b>Probability</b></span>
            <span>{prob:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Top Factors
    st.markdown("#### 🔑 Top Influencing Factors")

    for f in top_features[:3]:
            clean_name = f.replace("_", " ").replace("male", "").title()
            st.markdown(f"""
            <div style="
                background:#eef2ff;
                padding:8px 12px;
                border-radius:8px;
                margin-bottom:6px;
            ">
                {clean_name}
            </div>
            """, unsafe_allow_html=True)

        # Health Indicators
    st.markdown("#### 🩺 Health Indicators")

    st.markdown(f"""
        <div style="display:flex; justify-content:space-between;">
            <span>HbA1c</span><b>{hba1c}</b>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span>Glucose</span><b>{glucose}</b>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span>BMI</span><b>{bmi}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # DOWNLOAD REPORT
    # ----------------------
    report = generate_report(input_data, prob, risk_level, top_features)

    st.download_button(
        "📥 Download Report",
        report,
        file_name="diabetes_report.txt"
    )