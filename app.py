import streamlit as st
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EOR Screening Assistant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist tech aesthetic, rounded cards, and touch-friendly targets
st.markdown("""
    <style>
    /* Background & Container Styling */
    .stApp {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    /* Input Field Customization */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        min-height: 48px !important;
        border-color: #CBD5E1 !important;
    }
    
    /* Primary CTA Button */
    .stButton > button {
        background-color: #0F52BA;
        color: #FFFFFF;
        border-radius: 8px;
        height: 52px;
        font-size: 16px;
        font-weight: 700;
        width: 100%;
        border: none;
        transition: all 0.2s ease-in-out;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #0B3C8A;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(15, 82, 186, 0.25);
    }

    /* Metric Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 20px;
        font-weight: 700;
    }
    
    /* Custom Section Headers */
    .section-header {
        color: #0F172A;
        font-size: 18px;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & DATA MAPPINGS
# -----------------------------------------------------------------------------
EOR_TARGET_MAP = {
    0: "Hybrid Gas-Chemical",
    1: "Immiscible Gas Injection",
    2: "Miscible Gas Injection",
    3: "Secondary Waterflooding",
    4: "WAG Injection"
}

# Helper function to render color-coded eligibility badges
def render_badge(label, is_eligible):
    if is_eligible:
        bg, color, icon, text = "#F0FDF4", "#16A34A", "✓", "Eligible"
    else:
        bg, color, icon, text = "#FEF2F2", "#DC2626", "✕", "Ineligible"
    
    return f"""
    <div style="background-color: {bg}; border: 1px solid {color}33; padding: 10px 14px; 
                border-radius: 8px; margin-bottom: 8px; display: flex; 
                justify-content: space-between; align-items: center;">
        <span style="color: #334155; font-size: 14px; font-weight: 500;">{label}</span>
        <span style="color: {color}; font-weight: 700; font-size: 13px;">{icon} {text}</span>
    </div>
    """

# -----------------------------------------------------------------------------
# 3. DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.title("💧 EOR Screening Assistant")
st.caption("Enhanced Oil Recovery target classification & hydro-dynamic proxy analyzer")

st.divider()

# -----------------------------------------------------------------------------
# 4. INPUT FORM SECTIONS
# -----------------------------------------------------------------------------

# --- SECTION 1: RESERVOIR PARAMETERS ---
st.markdown('<div class="section-header">🪨 Reservoir Parameters</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    lithology = st.selectbox("Lithology", ["Sandstone", "Carbonate"], index=0)
    permeability = st.number_input("Permeability (mD)", min_value=0.01, value=120.50, step=1.0)

with col2:
    porosity = st.number_input("Porosity (%)", min_value=1.0, max_value=50.0, value=18.20, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=200.0, value=75.00, step=0.5)

# --- SECTION 2: FLUID PROPERTIES ---
st.markdown('<div class="section-header">🧪 Fluid Properties</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    viscosity = st.number_input("Oil Viscosity (cP)", min_value=0.01, value=2.40, step=0.1)

with col4:
    api = st.number_input("API Gravity (°API)", min_value=5.0, max_value=70.0, value=34.10, step=0.1)

st.divider()

# -----------------------------------------------------------------------------
# 5. DYNAMIC PROXY CALCULATIONS
# -----------------------------------------------------------------------------
# Calculating proxies (Mobility Proxy, RQI, FZI) dynamically from inputs
mobility_proxy = permeability / viscosity if viscosity > 0 else 0.0
rqi = 0.0314 * np.sqrt(permeability / porosity) if porosity > 0 else 0.0
phi_z = (porosity / 100) / (1 - (porosity / 100)) if porosity < 100 else 0.01
fzi = rqi / phi_z if phi_z > 0 else 0.0

# Dynamic Screening Rule Checks (Example EOR Feasibility Logic)
pass_miscible = bool(api > 30 and viscosity < 10 and temperature > 50)
pass_immiscible = bool(api > 15 and viscosity < 100)
pass_wag = bool(permeability > 10 and porosity > 10)
pass_chemical = bool(viscosity < 50 and temperature < 90)
pass_hybrid = bool(pass_chemical and pass_miscible)
pass_waterflood = bool(permeability > 5 and viscosity < 200)

# -----------------------------------------------------------------------------
# 6. MODEL EXECUTION & DISPLAY
# -----------------------------------------------------------------------------
if st.button("⚡ Screen EOR Technique"):
    st.markdown("---")
    st.subheader("📊 Analysis & Recommendation Output")
    
    # Simple Mock XGBoost Prediction Selection based on Rules
    if pass_miscible and pass_wag:
        predicted_class_id = 2  # Miscible Gas Injection
    elif pass_hybrid:
        predicted_class_id = 0  # Hybrid Gas-Chemical
    elif pass_wag:
        predicted_class_id = 4  # WAG Injection
    elif pass_immiscible:
        predicted_class_id = 1  # Immiscible Gas Injection
    else:
        predicted_class_id = 3  # Secondary Waterflooding
        
    target_technique = EOR_TARGET_MAP.get(predicted_class_id, "Unknown Target")
    
    # Recommendation Result Banner
    st.success(f"**Recommended Primary EOR Method:** {target_technique}")
    
    # Outputs: Section A (Advanced Proxies)
    st.markdown("##### Hydrocarbon Proxies")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mobility Proxy</div>
                <div class="metric-value">{mobility_proxy:.3f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RQI (μm)</div>
                <div class="metric-value">{rqi:.3f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FZI (μm)</div>
                <div class="metric-value">{fzi:.3f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Outputs: Section B (Feasibility Checks)
    st.markdown("##### Feasibility Criteria Checks")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown(render_badge("Pass Miscible", pass_miscible), unsafe_allow_html=True)
        st.markdown(render_badge("Pass Immiscible", pass_immiscible), unsafe_allow_html=True)
        st.markdown(render_badge("Pass WAG", pass_wag), unsafe_allow_html=True)
        
    with f_col2:
        st.markdown(render_badge("Pass Chemical", pass_chemical), unsafe_allow_html=True)
        st.markdown(render_badge("Pass Hybrid", pass_hybrid), unsafe_allow_html=True)
        st.markdown(render_badge("Pass Waterflood", pass_waterflood), unsafe_allow_html=True)