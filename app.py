import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import plotly.graph_objects as go

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = Path("models/best_model.h5")
SCALER_PATH = Path("models/scaler.pkl")

st.set_page_config(
    page_title="NASA Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------
# LOAD MODEL & SCALER
# ------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()

# ------------------------------
# HEADER
# ------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #0B3D91, #1E90FF);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
    ">
    <h1 style="font-size:48px;margin:0;">🪐 NASA Exoplanet Detection</h1>
    <p style="font-size:20px;margin:0;">
    Predict whether a transit signal corresponds to a real exoplanet using a high-performance 1D CNN.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("High-accuracy 1D CNN trained on synthetic exoplanet transit data.")
    st.markdown("---")
    st.subheader("💡 Tips")
    st.write(
        "- Use realistic feature values.\n"
        "- Probability > 50% → Likely an Exoplanet.\n"
        "- Probability ≤ 50% → Likely Not an Exoplanet."
    )
    st.markdown("---")
    st.subheader("📁 Resources")
    st.write(
        "[Project Repo](https://github.com/aysh34/2025-NASA-Space-Apps-Challenge) | [NASA Data](https://exoplanetarchive.ipac.caltech.edu/)"
    )

# ------------------------------
# USER INPUT FORM
# ------------------------------
st.subheader("Enter Observational Features")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tce_period = st.number_input(
            "Orbital Period (days)", min_value=0.0, value=10.0, step=0.1
        )
        tce_time0bk = st.number_input(
            "Transit Epoch (BKJD)", min_value=0.0, value=137.0, step=0.1
        )
        tce_duration = st.number_input(
            "Transit Duration (hrs)", min_value=0.0, value=6.0, step=0.1
        )

    with col2:
        tce_depth = st.number_input(
            "Transit Depth", min_value=0.0, value=500.0, step=1.0
        )
        tce_prad = st.number_input(
            "Planet Radius (Earth Radii)", min_value=0.0, value=2.0, step=0.1
        )
        tce_sma = st.number_input(
            "Semi-Major Axis", min_value=0.0, value=0.2, step=0.01
        )

    with col3:
        tce_impact = st.number_input(
            "Impact Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        tce_model_snr = st.number_input(
            "Model SNR", min_value=0.0, value=20.0, step=0.1
        )

    submitted = st.form_submit_button("Predict Exoplanet")

# ------------------------------
# PREDICTION
# ------------------------------
if submitted:
    features = np.array(
        [
            [
                tce_period,
                tce_time0bk,
                tce_duration,
                tce_depth,
                tce_prad,
                tce_sma,
                tce_impact,
                tce_model_snr,
            ]
        ]
    )
    features_scaled = scaler.transform(features)
    features_cnn = features_scaled.reshape(
        features_scaled.shape[0], features_scaled.shape[1], 1
    )
    prob = float(model.predict(features_cnn, verbose=0))

    # --------------------------
    # RADIAL PROBABILITY GAUGE
    # --------------------------
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "darkblue"},
                "bar": {"color": "#1E90FF"},
                "bgcolor": "#E8E8E8",
                "steps": [
                    {"range": [0, 50], "color": "#FF6F61"},
                    {"range": [50, 100], "color": "#3CB371"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
            },
        )
    )
    fig.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0))

    st.subheader("Prediction Result")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Probability > 50% → Likely an Exoplanet\n"
        "Probability ≤ 50% → Likely Not an Exoplanet"
    )

    # --------------------------
    # TRANSIT SIGNAL SIMULATION
    # --------------------------
    st.subheader("Transit Signal Simulation")
    time = np.linspace(0, 6 * np.pi, 100)
    signal = np.sin(time) * prob + np.random.normal(0, 0.05, 100)
    st.line_chart(pd.DataFrame({"Time": time, "Signal": signal}).set_index("Time"))

    # --------------------------
    # FEATURE IMPORTANCE PLACEHOLDER
    # --------------------------
    st.subheader("Feature Contribution (Simulated)")
    features_names = [
        "Period",
        "Epoch",
        "Duration",
        "Depth",
        "Radius",
        "SMA",
        "Impact",
        "SNR",
    ]
    contribution = np.random.rand(8) * prob
    fig2 = go.Figure(
        go.Barpolar(
            r=contribution,
            theta=[i * 45 for i in range(8)],
            width=[40] * 8,
            marker_color=["#0B3D91"] * 8,
            marker_line_color="black",
            marker_line_width=1,
        )
    )
    fig2.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# SAMPLE DATA EXPANDER
# ------------------------------
with st.expander("See Sample Dataset"):
    st.write(
        pd.DataFrame(
            {
                "tce_period": [3.2, 12.5],
                "tce_time0bk": [137.1, 138.5],
                "tce_duration": [5.1, 8.2],
                "tce_depth": [800, 350],
                "tce_prad": [3.5, 1.2],
                "tce_sma": [0.21, 0.5],
                "tce_impact": [0.1, 0.6],
                "tce_model_snr": [22.0, 15.0],
                "av_training_set": ["PC", "AFP"],
            }
        )
    )
