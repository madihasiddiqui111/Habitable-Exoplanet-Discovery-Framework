import numpy as np
import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
import google.generativeai as genai
from time import sleep

# Optional: Header image
st.image("galaxy.jpg", use_container_width =True)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #0f1a2b;
        color: white;
    }
    .stTextInput > div > div > input {
        color: black;
    }
    .stMarkdown h1 {
        color: #00c3ff;
    }
    .metric-label {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🪐 Planet Habitability Explorer")
st.markdown("<hr style='border: 1px solid #000000;'>", unsafe_allow_html=True) # Border after title
st.markdown("Select an exoplanet to discover its potential to support life 🌌")

# Load dataset
dt = pd.read_csv('preprocessed_dataset.csv')
selected_name = st.selectbox("🔭 Select a planet name:", dt['P_NAME'].unique())
st.markdown("<hr style='border: 1px solid #000000;'>", unsafe_allow_html=True) # Border after selectbox

if selected_name:
    filtered_dt = dt[dt['P_NAME'] == selected_name]
    row = filtered_dt.iloc[0]

    esi = row.get('ESI')
    ar = row.get('AR')
    lts = row.get('Long_Term_Stability')

    # Calculate Habitability Index
    if esi is not None and ar is not None and lts is not None:
        scaled_ar = ar / 100.0
        scaled_lts = lts / 100.0
        prediction = (0.6 * esi + 0.2 * scaled_ar + 0.2 * scaled_lts)
    else:
        prediction = "N/A (Required features missing)"

    # LLM Prompt Generator
    def generate_llm_prompt_from_row(row, prediction, esi, ar, lts):
        prompt_lines = [
            f"Details about the Habitability Index prediction for {row['P_NAME']}:\n\n",
            "The Habitability Index is calculated based on three key features:\n",
            "Earth Similarity Index (ESI), Atmospheric Retention (AR), and Long Term Stability.\n\n",
            "ESI ranges from 0 to 1.\n",
            "AR ranges from 0 to 100.\n",
            "Long Term Stability ranges from 0 to 100.\n\n",
            "The Habitability Index is calculated as the average of ESI and the scaled values of AR and Long Term Stability (scaled to a range of 0 to 1 by dividing by 100). The formula is:\n",
            "Habitability Index = (ESI + (AR / 100) + (Long Term Stability / 100)) / 3\n\n",
            f"ESI Value: {esi:.2f}\n",
            f"AR Value: {scaled_ar:.2f}\n",
            f"Long Term Stability Value: {scaled_lts:.2f}\n",
            f"\nPrediction (Habitability Index): {prediction:.2f}\n",
            "\nPlease provide a detailed explanation of this Habitability Index based on the given values."
        ]
        return "\n".join(prompt_lines)

    # Generate explanation
    if isinstance(prediction, float):
        prompt = generate_llm_prompt_from_row(row, prediction, esi, ar, lts)
        genai.configure(api_key="API-Key")  # Replace with your Gemini API Key
        model_llm = genai.GenerativeModel("gemini-1.5-flash")
        with st.spinner("Thinking like a cosmic AI..."):
            sleep(2)
            response = model_llm.generate_content(prompt).text
    else:
        response = "Could not calculate Habitability Index due to missing features."

    # Display key info with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌍 Habitability Index", f"{prediction:.2f}" if isinstance(prediction, float) else "N/A")
    with col2:
        st.metric("🔵 Earth Similarity Index (ESI)", f"{esi:.2f}" if esi else "N/A")
    with col3:
        st.metric("🌀 Atmospheric Retention (AR)", f"{scaled_ar:.2f}" if ar else "N/A")

    col4, col5 = st.columns(2)
    with col4:
        st.metric("🧭 Long Term Stability", f"{scaled_lts:.2f}" if lts else "N/A")
    with col5:
        if isinstance(prediction, float):
            st.markdown("📊 Habitability Score Progress")
            st.progress(min(prediction, 1.0))

    # Categorize habitability
    habitability_category = ""
    if isinstance(prediction, float):
        if prediction > 0.56 and esi > 0.8 and scaled_lts > 0.8:
            habitability_category = "**Potentially Habitable**"
        elif prediction >= 0.43 and prediction <= 0.74:
            habitability_category = "**Marginally Habitable**"
        else:
            habitability_category = "**Non Habitable**"

    st.markdown(f"### 🧬 Habitability Assessment: {habitability_category}")
    st.markdown("<hr style='border: 1px solid #000000;'>", unsafe_allow_html=True) # Border after Habitability Assessment

        # Gauge Chart Visualization
    if all(isinstance(v, float) for v in [esi, ar, lts]):
        st.markdown("### 🧭 Planetary Feature Gauges")

        col1, col2, col3 = st.columns(3)

        with col1:
            gauge1 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=esi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🌍 ESI"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "#ffcccc"},
                        {'range': [0.4, 0.7], 'color': "#fff3cd"},
                        {'range': [0.7, 1], 'color': "#d4edda"}
                    ],
                }
            ))
            st.plotly_chart(gauge1, use_container_width=True)

        with col2:
            gauge2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=ar / 100.0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🌀 AR (Scaled)"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#ff7f0e"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "#ffcccc"},
                        {'range': [0.4, 0.7], 'color': "#fff3cd"},
                        {'range': [0.7, 1], 'color': "#d4edda"}
                    ],
                }
            ))
            st.plotly_chart(gauge2, use_container_width=True)

        with col3:
            gauge3 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=lts / 100.0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🧭 LTS (Scaled)"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#2ca02c"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "#ffcccc"},
                        {'range': [0.4, 0.7], 'color': "#fff3cd"},
                        {'range': [0.7, 1], 'color': "#d4edda"}
                    ],
                }
            ))
            st.plotly_chart(gauge3, use_container_width=True)
        st.markdown("<hr style='border: 1px solid #000000;'>", unsafe_allow_html=True) # Border after Gauges

    # LLM Explanation
    st.markdown("### 🧠 Interpretation")
    with st.expander("Click to view explanation"):
        st.write(response)
