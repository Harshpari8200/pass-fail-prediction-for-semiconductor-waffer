#!/usr/bin/env python3
"""
Semiconductor Wafer Pass/Fail Prediction - Streamlit Web App
This app allows users to input wafer parameters and get Pass/Fail predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Semiconductor Wafer Pass/Fail Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    try:
        model = joblib.load('models/wafer_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names, True
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please ensure you have run 'python main.py' to train the model first.")
        return None, None, None, False

def main():
    """Main Streamlit application"""

    # Title and description
    st.title("🔬 Semiconductor Wafer Pass/Fail Prediction")
    st.markdown("""
    This application predicts whether a semiconductor wafer will **Pass** or **Fail** 
    quality control testing based on manufacturing parameters.
    """)

    # Load model artifacts
    model, scaler, feature_names, model_loaded = load_model_artifacts()

    if not model_loaded:
        st.stop()

    # Sidebar for model information
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("""
    **Model Type**: Random Forest Classifier
    **Accuracy**: 99.0%
    **Features**: 9 manufacturing parameters
    **Training Data**: 499 wafer samples
    """)

    # Feature descriptions
    feature_descriptions = {
        'Wafer_Thickness_um': 'Wafer thickness in micrometers (typical: 700-750 μm)',
        'Surface_Defect_Count': 'Number of surface defects per cm² (target: ≤10)',
        'Resistance_Ohms': 'Electrical resistance in Ohms (typical: 10-50 Ω)',
        'Voltage_V': 'Operating voltage in Volts (typical: 3.2-3.4 V)',
        'Temperature_C': 'Process temperature in Celsius (typical: 900-1100°C)',
        'Process_Time_min': 'Manufacturing time in minutes (typical: 120-180 min)',
        'Chemical_Concentration_ppm': 'Chemical impurity level in ppm (target: ≤5 ppm)',
        'Surface_Roughness_nm': 'Surface roughness in nanometers (target: ≤5 nm)',
        'Pattern_Alignment_Error_nm': 'Alignment error in nanometers (target: ≤50 nm)'
    }

    # Input form
    st.header("📝 Enter Wafer Parameters")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Input fields
    input_data = {}

    with col1:
        st.subheader("Physical Parameters")
        input_data['Wafer_Thickness_um'] = st.number_input(
            "Wafer Thickness (μm)",
            min_value=650.0, max_value=800.0, value=725.0, step=0.1,
            help=feature_descriptions['Wafer_Thickness_um']
        )

        input_data['Surface_Defect_Count'] = st.number_input(
            "Surface Defect Count",
            min_value=0, max_value=50, value=3, step=1,
            help=feature_descriptions['Surface_Defect_Count']
        )

        input_data['Surface_Roughness_nm'] = st.number_input(
            "Surface Roughness (nm)",
            min_value=0.5, max_value=20.0, value=3.0, step=0.1,
            help=feature_descriptions['Surface_Roughness_nm']
        )

        input_data['Pattern_Alignment_Error_nm'] = st.number_input(
            "Pattern Alignment Error (nm)",
            min_value=5.0, max_value=150.0, value=25.0, step=0.1,
            help=feature_descriptions['Pattern_Alignment_Error_nm']
        )

    with col2:
        st.subheader("Electrical & Process Parameters")
        input_data['Resistance_Ohms'] = st.number_input(
            "Resistance (Ω)",
            min_value=0.1, max_value=150.0, value=30.0, step=0.1,
            help=feature_descriptions['Resistance_Ohms']
        )

        input_data['Voltage_V'] = st.number_input(
            "Voltage (V)",
            min_value=2.5, max_value=4.0, value=3.3, step=0.01,
            help=feature_descriptions['Voltage_V']
        )

        input_data['Temperature_C'] = st.number_input(
            "Process Temperature (°C)",
            min_value=750.0, max_value=1250.0, value=1000.0, step=1.0,
            help=feature_descriptions['Temperature_C']
        )

        input_data['Process_Time_min'] = st.number_input(
            "Process Time (min)",
            min_value=80.0, max_value=220.0, value=150.0, step=0.1,
            help=feature_descriptions['Process_Time_min']
        )

        input_data['Chemical_Concentration_ppm'] = st.number_input(
            "Chemical Concentration (ppm)",
            min_value=0.1, max_value=30.0, value=2.0, step=0.1,
            help=feature_descriptions['Chemical_Concentration_ppm']
        )

    # Prediction button
    st.markdown("---")

    if st.button("🔍 Predict Wafer Quality", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])

            # Ensure the order matches training data
            input_df = input_df[feature_names]

            # Scale the input
            input_scaled = scaler.transform(input_df)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")

            # Create result columns
            result_col1, result_col2, result_col3 = st.columns([2, 1, 2])

            with result_col2:
                if prediction == 'Pass':
                    st.success("✅ PASS")
                    confidence = probability[1] * 100  # Pass probability
                else:
                    st.error("❌ FAIL")
                    confidence = probability[0] * 100  # Fail probability

                st.metric("Confidence", f"{confidence:.1f}%")

            # Detailed probabilities
            st.subheader("📊 Prediction Probabilities")
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.metric("❌ Fail Probability", f"{probability[0]:.3f} ({probability[0]*100:.1f}%)")

            with prob_col2:
                st.metric("✅ Pass Probability", f"{probability[1]:.3f} ({probability[1]*100:.1f}%)")

            # Input summary
            st.subheader("📋 Input Parameters Summary")
            summary_df = pd.DataFrame([input_data]).T
            summary_df.columns = ['Value']
            summary_df.index.name = 'Parameter'
            st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p> Semiconductor Wafer Quality Control System </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
