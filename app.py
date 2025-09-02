import streamlit as st
import pickle
import numpy as np
import os

# -------------------
# CONFIGURATION
# -------------------
st.set_page_config(
    page_title="Order Fulfillment Estimator",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fafb;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# LOAD MODEL SAFELY
# -------------------
model_path = "fulfillment_time_model.pkl"
model = None

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.error(f"❌ Model file not found: `{model_path}`. Please make sure it’s in the same folder as this app.")

# -------------------
# TITLE & DESCRIPTION
# -------------------
st.title("📦 Order Fulfillment Time Estimator")
st.write("Easily estimate how long it will take to fulfill an order based on key features.")

st.markdown("---")

# -------------------
# INPUT FORM
# -------------------
col1, col2, col3 = st.columns(3)
with col1:
    feature1 = st.number_input("📏 Item Count", value=0.0, step=0.1)
with col2:
    feature2 = st.number_input("📍 Distance (km)", value=0.0, step=0.1)
with col3:
    feature3 = st.number_input("⚡ Priority", value=0.0, step=0.1)

col4, col5 = st.columns(2)
with col4:
    feature4 = st.number_input("🏭 Warehouse Load (%)", value=0.0, step=0.1)
with col5:
    feature5 = st.number_input("🚚 Delivery Method (0/1)", value=0.0, step=0.1)

# -------------------
# PREDICTION
# -------------------
if st.button("🚀 Estimate Time"):
    if model is not None:
        try:
            features = np.array([[feature1, feature2, feature3, feature4, feature5]])
            prediction = model.predict(features)
            st.success(f"⏳ Estimated fulfillment time: **{prediction[0]:.2f} hours**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.warning("⚠️ Model is not loaded. Predictions are disabled until a compatible model is provided.")

# -------------------
# FOOTER
# -------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
