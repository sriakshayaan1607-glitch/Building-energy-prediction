
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "cooling load dataset.xlsx"
MODEL_FILE = BASE / "advanced_results" / "Deep_ANN.keras"
SCALER_FILE = BASE / "advanced_results" / "advanced_scaler.pkl"

FEATURES = [
    "Relative compactness",
    "Surface area",
    "Wall area",
    "Roof area",
    "Overall height",
    "Orientation",
    "Glazing area",
    "Glazing distribution",
]

st.set_page_config(
    page_title="Building Energy Predictor",
    page_icon="🏢",
    layout="wide",
)

@st.cache_resource
def load_assets():
    data = pd.read_excel(DATA_FILE)
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    scaler = joblib.load(SCALER_FILE)
    return data, model, scaler

st.title("🏢 Building Energy Prediction")
st.subheader("Deep Learning-Based Heating and Cooling Load Estimation")
st.write(
    "Enter building design parameters to estimate heating and cooling loads "
    "using a trained Artificial Neural Network (ANN)."
)

try:
    data, model, scaler = load_assets()
except Exception as exc:
    st.error("The app could not load its dataset, model, or scaler.")
    st.code(str(exc))
    st.info(
        "Check that the dataset is in data/ and that Deep_ANN.keras "
        "and advanced_scaler.pkl are in advanced_results/."
    )
    st.stop()

if not all(c in data.columns for c in [f"X{i}" for i in range(1, 9)]):
    st.error("The dataset must contain columns X1 through X8.")
    st.stop()

ranges = {
    f"X{i}": (float(data[f"X{i}"].min()), float(data[f"X{i}"].max()))
    for i in range(1, 9)
}

def building_inputs(prefix):
    st.markdown("#### Building design parameters")
    values = []

    labels = [
        "Relative compactness",
        "Surface area",
        "Wall area",
        "Roof area",
        "Overall height",
        "Orientation",
        "Glazing area",
        "Glazing distribution",
    ]

    for i, label in enumerate(labels, start=1):
        col = f"X{i}"
        low, high = ranges[col]

        if i in (6, 8):
            choices = sorted(data[col].dropna().unique().tolist())
            value = st.selectbox(
                label,
                choices,
                key=f"{prefix}_{col}",
            )
        else:
            value = st.slider(
                label,
                min_value=low,
                max_value=high,
                value=float(data[col].median()),
                key=f"{prefix}_{col}",
            )

        values.append(float(value))

    return values

def predict(values):
    row = pd.DataFrame([values], columns=[f"X{i}" for i in range(1, 9)])
    scaled = scaler.transform(row)
    prediction = np.asarray(model.predict(scaled, verbose=0)).reshape(-1)

    if len(prediction) != 2:
        raise ValueError(
            "Expected the model to predict two outputs: heating and cooling load."
        )

    return float(prediction[0]), float(prediction[1])

tab1, tab2 = st.tabs(["Single prediction", "Compare two scenarios"])

with tab1:
    with st.form("single_form"):
        inputs = building_inputs("single")
        submitted = st.form_submit_button("Predict energy loads")

    if submitted:
        try:
            heating, cooling = predict(inputs)
            c1, c2 = st.columns(2)
            c1.metric("Predicted Heating Load", f"{heating:.2f}")
            c2.metric("Predicted Cooling Load", f"{cooling:.2f}")
            st.caption(
                "Predictions use the dataset's target units. "
                "These are academic estimates, not certified engineering results."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab2:
    left, right = st.columns(2)

    with left:
        st.markdown("### Scenario A")
        with st.form("scenario_a_form"):
            a_inputs = building_inputs("scenario_a")
            a_submit = st.form_submit_button("Predict Scenario A")

    with right:
        st.markdown("### Scenario B")
        with st.form("scenario_b_form"):
            b_inputs = building_inputs("scenario_b")
            b_submit = st.form_submit_button("Predict Scenario B")

    if a_submit or b_submit:
        try:
            if a_submit:
                st.session_state["scenario_a"] = predict(a_inputs)
            if b_submit:
                st.session_state["scenario_b"] = predict(b_inputs)

            a = st.session_state.get("scenario_a")
            b = st.session_state.get("scenario_b")

            if a is not None and b is not None:
                comparison = pd.DataFrame(
                    {
                        "Scenario A": [a[0], a[1]],
                        "Scenario B": [b[0], b[1]],
                    },
                    index=["Heating Load", "Cooling Load"],
                )
                st.markdown("### Results comparison")
                st.dataframe(comparison)
                st.bar_chart(comparison)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

st.divider()
st.caption(
    "Academic deep learning project | Model: Deep ANN | "
    "Results depend on the training data and model."
)
