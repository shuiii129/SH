import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import google.generativeai as genai
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time

# --- 1. CONFIGURATION ---
MODEL_FILE = "amanpay_model.keras"
SCALER_FILE = "scaler.pkl"
FIREBASE_KEY = "key.json"  # Ensure you uploaded this file!

# !!! PASTE YOUR GOOGLE GEMINI API KEY HERE !!!
GOOG_API_KEY = "AIzaSyA4j5xl1lkGEKx3LTTYsTJMCs9q6FNpDkE"

# --- 2. SETUP FIREBASE ---
# We use a "Try/Except" block to prevent errors when Streamlit reloads
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_KEY)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://aipay-61b05-default-rtdb.asia-southeast1.firebasedatabase.app/'
            # ^^^ IMPORTANT: REPLACE THIS URL WITH YOUR OWN FIREBASE URL IF IT FAILS ^^^
        })
        st.toast("🔥 Firebase Connected!", icon="☁️")
    except Exception as e:
        st.warning(f"Firebase not connected: {e}")

# --- 3. AUTO-DETECT GEMINI MODEL ---
gemini_active = False
model_gemini = None
try:
    if GOOG_API_KEY.startswith("AIza"):
        genai.configure(api_key=GOOG_API_KEY)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        chosen_model = next((m for m in available_models if "gemini" in m), available_models[0] if available_models else None)
        if chosen_model:
            model_gemini = genai.GenerativeModel(chosen_model)
            gemini_active = True
except:
    gemini_active = False

# --- 4. LOAD BRAIN ---
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE): return None, None
    model = tf.keras.models.load_model(MODEL_FILE)
    with open(SCALER_FILE, 'rb') as f: scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_resources()
    if model is None: st.stop()
except: st.stop()

# --- 5. UI LAYOUT ---
st.set_page_config(page_title="AmanPay AI", page_icon="🛡️", layout="wide")
st.title("🛡️ AmanPay AI")
st.markdown("### Securing the Digital Economy for Malaysia’s Underserved")
st.markdown("---")

# --- 6. SIDEBAR INPUTS ---
st.sidebar.header("📝 Transaction Simulator")
amount = st.sidebar.number_input("Amount (RM)", min_value=1.0, value=20.0, step=10.0)
hour = st.sidebar.slider("Time of Day (24h)", 0, 23, 14)
st.sidebar.subheader("📍 Location Details")
lat = st.sidebar.number_input("Latitude", value=3.1408, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=101.6932, format="%.4f")
btn = st.sidebar.button("🚀 Analyze Transaction", type="primary")

# --- 7. MAIN LOGIC ---
if btn:
    # Scale & Predict
    input_data = pd.DataFrame([[amount, hour, lat, lon]], columns=['amount', 'hour', 'lat', 'lon'])
    input_scaled = scaler.transform(input_data)
    reconstructed = model.predict(input_scaled)
    error_score = np.mean(np.power(input_scaled - reconstructed, 2), axis=1)[0]
    is_fraud = error_score > 0.5

    # === FIREBASE LOGGING ===
    # This sends the data to the cloud instantly
    try:
        ref = db.reference("/transactions")
        ref.push({
            "amount": amount,
            "hour": hour,
            "risk_score": float(error_score),
            "status": "FRAUD" if is_fraud else "SAFE",
            "timestamp": time.time()
        })
        st.toast("Data synced to Cloud Database", icon="✅")
    except Exception as e:
        print(f"Firebase Error: {e}")

    # === DISPLAY RESULTS ===
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Analysis Result")
        if is_fraud:
            st.error(f"🚨 FRAUD DETECTED!")
            st.metric("Risk Score", f"{error_score:.4f}", "High Risk", delta_color="inverse")

            st.markdown("#### 🤖 AI Guardian Explanation:")
            if gemini_active:
                with st.spinner("Analyzing context..."):
                    try:
                        prompt = f"Act as a security bot. A user spent RM {amount} at hour {hour}. Location: {lat}, {lon}. Risk Score: {error_score:.2f}. Explain professionally in English why this is blocked in 1 sentence."
                        response = model_gemini.generate_content(prompt)
                        st.info(response.text)
                    except: st.error("AI Busy.")
        else:
            st.success(f"✅ Transaction Approved")
            st.metric("Risk Score", f"{error_score:.4f}", "Safe")

    with col2:
        st.subheader("🌍 Location Tracker")
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

# --- 8. LIVE HISTORY SECTION ---
st.markdown("---")
st.subheader("☁️ Live Cloud Database Records")
try:
    # Read the last 5 transactions from Firebase
    ref = db.reference("/transactions")
    snapshot = ref.order_by_key().limit_to_last(5).get()

    if snapshot:
        # Convert Firebase JSON to a nice Table
        data_list = []
        for key, val in snapshot.items():
            data_list.append(val)
        df_history = pd.DataFrame(data_list)
        # Reorder columns for neatness
        st.dataframe(df_history[['status', 'amount', 'risk_score', 'hour']], use_container_width=True)
    else:
        st.write("No records in cloud yet.")
except:
    st.write("Connect Firebase to see history.")
