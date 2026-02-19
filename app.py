import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'lgb_flood_model.txt'

# List of original features (20 features)
FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", 
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
]

# ==========================================
# LOGIC
# ==========================================

@st.cache_resource
def load_model():
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Did you run train_save.py first?")
        return None

def add_row_stats(df):
    """Replicates the feature engineering used during training"""
    df = df.copy()
    num = df.select_dtypes(include=[np.number])
    df["row_mean"] = num.mean(axis=1)
    df["row_std"]  = num.std(axis=1)
    df["row_min"]  = num.min(axis=1)
    df["row_max"]  = num.max(axis=1)
    df["row_sum"]  = num.sum(axis=1)
    return df

# ==========================================
# UI LAYOUT
# ==========================================
st.set_page_config(page_title="Flood Predictor", layout="wide")

st.title("🌊 Flood Probability Predictor")
st.markdown("Adjust the environmental and infrastructural factors below to see the predicted flood risk.")

# Load Model
model = load_model()

if model:
    # --- Input Section ---
    st.sidebar.header("Feature Configuration")
    
    # Create a dictionary to hold user inputs
    input_data = {}

    # Organize sliders into tabs for better UX (20 sliders is a lot for one list)
    tab1, tab2, tab3 = st.tabs(["🌧️ Environment", "🏗️ Infrastructure", "🏙️ Social & Political"])

    with tab1:
        st.subheader("Environmental Factors")
        input_data['MonsoonIntensity'] = st.slider("Monsoon Intensity", 0, 20, 5)
        input_data['ClimateChange'] = st.slider("Climate Change Impact", 0, 20, 5)
        input_data['TopographyDrainage'] = st.slider("Topography Drainage", 0, 20, 5)
        input_data['RiverManagement'] = st.slider("River Management", 0, 20, 5)
        input_data['Deforestation'] = st.slider("Deforestation", 0, 20, 5)
        input_data['Siltation'] = st.slider("Siltation", 0, 20, 5)
        input_data['Landslides'] = st.slider("Landslides", 0, 20, 5)
        input_data['Watersheds'] = st.slider("Watersheds Health", 0, 20, 5)

    with tab2:
        st.subheader("Infrastructure Factors")
        input_data['DamsQuality'] = st.slider("Dams Quality", 0, 20, 5)
        input_data['DrainageSystems'] = st.slider("Drainage Systems", 0, 20, 5)
        input_data['DeterioratingInfrastructure'] = st.slider("Infrastructure Deterioration", 0, 20, 5)
        input_data['Encroachments'] = st.slider("Encroachments", 0, 20, 5)
        input_data['CoastalVulnerability'] = st.slider("Coastal Vulnerability", 0, 20, 5)

    with tab3:
        st.subheader("Social & Political Factors")
        input_data['Urbanization'] = st.slider("Urbanization", 0, 20, 5)
        input_data['AgriculturalPractices'] = st.slider("Agricultural Practices", 0, 20, 5)
        input_data['IneffectiveDisasterPreparedness'] = st.slider("Ineffective Disaster Prep", 0, 20, 5)
        input_data['PopulationScore'] = st.slider("Population Score", 0, 20, 5)
        input_data['WetlandLoss'] = st.slider("Wetland Loss", 0, 20, 5)
        input_data['InadequatePlanning'] = st.slider("Inadequate Planning", 0, 20, 5)
        input_data['PoliticalFactors'] = st.slider("Political Factors", 0, 20, 5)

    # --- Prediction Logic ---
    
    # 1. Convert dict to DataFrame (correct order)
    df_input = pd.DataFrame([input_data])
    
    # 2. Reorder columns to match original training data exactly
    df_input = df_input[FEATURES]
    
    # 3. Apply Feature Engineering
    df_fe = add_row_stats(df_input)

    # 4. Predict
    prediction = model.predict(df_fe)[0]

    # --- Results Display ---
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction Result")
        # Color coding based on risk
        color = "green"
        if prediction > 0.4: color = "orange"
        if prediction > 0.6: color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; border: 2px solid {color}; padding: 20px; border-radius: 10px;">
            <h1 style="color: {color};">{prediction:.2%}</h1>
            <p>Flood Probability</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Risk Analysis")
        if prediction < 0.4:
            st.success("✅ Low Risk: Conditions appear manageable.")
        elif prediction < 0.6:
            st.warning("⚠️ Moderate Risk: Caution advised. Infrastructure improvements recommended.")
        else:
            st.error("🚨 High Risk: Immediate attention required to mitigate potential flooding.")
            
        # Optional: Radar Chart to visualize inputs
        # Normalize inputs for simple visualization
        categories = list(input_data.keys())
        values = list(input_data.values())
        
        # Simple bar chart for top contributing factors (based on user input magnitude)
        st.caption("Input Feature Magnitudes")
        st.bar_chart(input_data)