import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.markdown(
    """
    <style>
    :root {
        --navy: #14213d;
        --blue: #246bfd;
        --pale: #f3f7ff;
        --text: #243047;
        --muted: #65738b;
        --border: #dce5f2;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f7faff 0%, #ffffff 45%, #f5f8ff 100%);
        color: var(--text);
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 980px; padding-top: 1.2rem; padding-bottom: 3rem; }
    .topbar {
        display:flex; align-items:center; justify-content:space-between; gap:12px;
        padding: 12px 0 22px; border-bottom: 1px solid var(--border); margin-bottom: 28px;
    }
    .brand { font-size: 1.15rem; font-weight: 800; color: var(--navy); letter-spacing:-.03em; }
    .brand span { color: var(--blue); }
    .top-pill { border:1px solid #cbdcff; color:#2456b7; background:#edf3ff;
        border-radius:999px; padding:7px 12px; font-size:.78rem; font-weight:700; }
    .hero {
        padding: 28px; border-radius: 24px; color: white;
        background: linear-gradient(125deg, #14213d 0%, #1e4e9d 58%, #3478ff 100%);
        box-shadow: 0 18px 45px rgba(27, 71, 145, .16); margin-bottom: 22px;
    }
    .hero-kicker { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem;
        font-weight:800; opacity:.82; margin-bottom:10px; }
    .hero h1 { font-size:clamp(1.8rem, 5vw, 2.8rem); line-height:1.08;
        margin:0 0 12px; color:white; letter-spacing:-.04em; }
    .hero p { max-width:650px; margin:0; line-height:1.65; color:#e6eeff; font-size:1rem; }
    .section-card {
        background:white; border:1px solid var(--border); border-radius:20px;
        padding:22px; margin:18px 0; box-shadow:0 8px 26px rgba(28, 55, 95, .045);
    }
    .section-title { color:var(--navy); font-size:1.15rem; font-weight:800; margin:0 0 5px; }
    .section-subtitle { color:var(--muted); font-size:.9rem; margin:0 0 18px; line-height:1.5; }
    .result-card { border-radius:20px; padding:22px; margin-top:20px; border:1px solid var(--border); background:#fff; }
    .result-good { border-left:6px solid #1e9b72; }
    .result-risk { border-left:6px solid #e39b24; }
    .result-heading { color:var(--navy); font-size:1.25rem; font-weight:800; margin-bottom:7px; }
    .result-copy { color:var(--muted); line-height:1.6; }
    .disclaimer { font-size:.8rem; color:#6a7689; line-height:1.55; padding:15px 0 0; }
    div[data-testid="stFormSubmitButton"] button {
        width:100%; border:0; border-radius:12px; padding:.78rem 1rem;
        background:linear-gradient(90deg,#246bfd,#397cff); color:white; font-weight:800;
    }
    div[data-testid="stFormSubmitButton"] button:hover { background:#174fc7; color:white; }
    label { font-weight:650 !important; color:#33415c !important; }
    @media (max-width: 640px) {
        .block-container { padding: .7rem 1rem 2rem; }
        .topbar { align-items:flex-start; flex-direction:column; margin-bottom:18px; }
        .hero { padding:22px 19px; border-radius:18px; }
        .section-card { padding:17px; border-radius:16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
      <div class="brand">Heart<span>Care</span></div>
      <div class="top-pill">HEALTH • PREDICTION</div>
    </div>
    <div class="hero">
      <h1>Heart disease prediction</h1>
      
    """,
    unsafe_allow_html=True,
)

DEFAULT_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_INFO = {
    "age": ("Age", 1, 120, 1, 50, "Age in years"),
    "sex": ("Sex (dataset code)", 0, 1, 1, 1, "Use the same coding as the training dataset"),
    "cp": ("Chest pain type", 0, 3, 1, 0, "Encoded category; match the training dataset"),
    "trestbps": ("Resting blood pressure", 50, 250, 1, 120, "Typically recorded in mm Hg"),
    "chol": ("Serum cholesterol", 80, 700, 1, 200, "Typically recorded in mg/dL"),
    "fbs": ("Fasting blood sugar (dataset code)", 0, 1, 1, 0, "Use the same coding as the training dataset"),
    "restecg": ("Resting ECG (dataset code)", 0, 2, 1, 0, "Encoded category; match the training dataset"),
    "thalach": ("Maximum heart rate", 40, 250, 1, 150, "Maximum heart rate achieved"),
    "exang": ("Exercise-induced angina (dataset code)", 0, 1, 1, 0, "Use the same coding as the training dataset"),
    "oldpeak": ("ST depression", 0.0, 10.0, 0.1, 1.0, "Numeric value from exercise test"),
    "slope": ("ST slope (dataset code)", 0, 2, 1, 1, "Encoded category; match the training dataset"),
    "ca": ("Number of major vessels", 0, 4, 1, 0, "Use the range represented in your dataset"),
    "thal": ("Thalassemia (dataset code)", 0, 3, 1, 2, "Encoded category; match the training dataset"),
}


MODEL_PATH = "heart_model_v20261013_160903.joblib"

def load_model(path):
    if not os.path.exists(path):
        return None, f"Model file not found: `{path}`"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Could not load the model: {exc}"

model, model_error = load_model(MODEL_PATH)


st.markdown(
    '<div class="section-card"><div class="section-title">Patient information</div>'
    '<div class="section-subtitle"></div>',
    unsafe_allow_html=True,
)

with st.form("patient_form"):
    values = {}
    feature_names = DEFAULT_FEATURES
    if model is not None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    for feature in feature_names:
        key = str(feature)
        if key in FEATURE_INFO:
            label, min_value, max_value, step, default_value, help_text = FEATURE_INFO[key]
        else:
            label, min_value, max_value, step, default_value, help_text = (
                key.replace("_", " ").title(), -1_000_000.0, 1_000_000.0, 0.1, 0.0,
                "Enter a value consistent with the model's training data."
            )
        values[key] = st.number_input(
            label, min_value=float(min_value), max_value=float(max_value),
            value=float(default_value), step=float(step), help=help_text, key=f"feature_{key}"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.error(
            f"The model could not be loaded. Put your trained model file at `{MODEL_PATH}` "
            "or set the HEART_MODEL_PATH environment variable."
        )
    else:
        try:
            input_df = pd.DataFrame([values], columns=feature_names)
            prediction = model.predict(input_df)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(getattr(model, "classes_", range(len(probabilities))))
                positive_index = 1 if len(probabilities) > 1 else 0
                probability = float(probabilities[positive_index])
                predicted_class = str(prediction)
                st.markdown('<div class="result-card result-risk">', unsafe_allow_html=True)
                st.markdown('<div class="result-heading">Prediction complete</div>', unsafe_allow_html=True)
                st.write(f"**Predicted class:** {predicted_class}")
                st.write(f"**Probability for class {classes[positive_index]}:** {probability:.1%}")
                st.markdown(
                    '<div class="result-copy">Probability is a model output, not a confirmed medical risk estimate. '
                    'Interpret it only after validating the model and its class mapping.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.success(f"Predicted class: {prediction}")
        except Exception as exc:
            st.error(
                "Prediction failed. Check that the input features, names, order, data types, "
                f"and category encoding match training. Details: {exc}"
            )


