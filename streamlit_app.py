import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
import joblib
from geo_features.dem_utils import get_elevation
from geopy.geocoders import Nominatim

# Load trained model
model = joblib.load("models/flood_model.pkl")
# ===============================
# Streamlit session defaults
# (prevents NameError before button click)
# ===============================
if "band_text" not in st.session_state:
    st.session_state.band_text = None
    st.session_state.band_color = None
    st.session_state.final_prediction = None

# Default elevation (before map click)
elevation = None
def risk_band(prob):
    if prob < 0.30:
        return "🟢 Low Risk", "green"
    elif prob < 0.55:
        return "🟡 Moderate Risk", "orange"
    else:
        return "🔴 High Risk", "red"

def elevation_risk_factor(elevation):
    if elevation is None:
        return 0.0
    elif elevation < 10:      # extreme coastal / floodplain
        return 0.30
    elif elevation < 50:      # coastal / delta
        return 0.20
    elif elevation < 200:     # river basin
        return 0.10
    else:                     # hills / plateau
        return -0.05
# ===============================
# Elevation band explanation
# ===============================
def elevation_band_label(elevation):
    if elevation is None:
        return "Unknown elevation", 0.0
    elif elevation < 10:
        return "Extreme Coastal Floodplain (0–10 m)", 0.30
    elif elevation < 50:
        return "Coastal / Delta Region (10–50 m)", 0.20
    elif elevation < 200:
        return "River Basin / Low Plateau (50–200 m)", 0.10
    else:
        return "Highland / Safe Elevation (>200 m)", -0.05

def flood_zone_risk(lat, lon, elevation):      #flood zone classification based on elevation and location 
    """
    Returns flood zone label and extra risk weight
    Rule-based geography (NO ML retrain)
    """
    if elevation is None:
        return "Unknown Flood Zone", 0.0

    # Coastal Odisha (very low elevation)
    if elevation < 10:
        return "Coastal Flood Zone", 0.25

    # Major river basins / deltas
    elif elevation < 50:
        return "River Basin Flood Zone", 0.15

    # Inland flood-prone plains
    elif elevation < 150:
        return "Inland Flood-Prone Zone", 0.08

    # Highlands
    else:
        return "Low Flood Risk Zone", 0.0



st.set_page_config(page_title="Flood Risk Prediction", layout="centered")
geolocator = Nominatim(user_agent="flood_risk_app")

st.markdown(
    """
    <style>
    div[data-testid="stIFrame"] {
        margin-bottom: -150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🌊 Flood Risk Prediction System")
st.write("Enter environmental and planning factors to predict flood probability.")



# Odisha center
st.subheader("📍 Select Location in Odisha")

odisha_lat, odisha_lon = 20.2961, 85.8245
m = folium.Map(location=[odisha_lat, odisha_lon], zoom_start=7)

folium.Marker(
    [odisha_lat, odisha_lon],
    tooltip="Odisha"
).add_to(m)

with st.container():
    map_data = st_folium(
        m,
        height=550,
        key="odisha_map"
    )

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        location = geolocator.reverse((lat, lon), language="en")
        place_name = location.address if location else "Unknown Location"


        st.success(f"📌 Selected Location: Lat {lat:.4f}, Lon {lon:.4f}")
        st.info(f"📍 Place: {place_name}")

        elevation = get_elevation(lat, lon)
       

        risk_factor = elevation_risk_factor(elevation)
        band_text, band_risk = elevation_band_label(elevation)
        zone_label, zone_risk = flood_zone_risk(lat, lon, elevation)
        st.session_state.flood_zone = zone_label

        st.info(
            f"🗻 Elevation: {elevation} m\n\n"
            f"🌍 Terrain Zone: {band_text}\n\n"
            f"🌊 Flood Zone: {zone_label}\n\n"
            f"⚠️ Elevation Risk Adjustment: {band_risk:+.2f}\n"
            f"⚠️ Flood Zone Risk Adjustment: {zone_risk:+.2f}"
        )







# Input fields
def user_input():
        # Feature descriptions for sliders
    feature_help = {
        "MonsoonIntensity": "Higher = heavier rainfall, Lower = weak monsoon",
        "TopographyDrainage": "Higher = poor drainage, Lower = good drainage",
        "RiverManagement": "Higher = weak river control, Lower = strong management",
        "Deforestation": "Higher = more forest loss, Lower = healthy forest cover",
        "Urbanization": "Higher = dense construction, Lower = rural/open areas",
        "ClimateChange": "Higher = severe climate impact, Lower = stable climate",
        "DamsQuality": "Higher = poor dam condition, Lower = strong dam safety",
        "Siltation": "Higher = more river silt, Lower = clean riverbeds",
        "AgriculturalPractices": "Higher = unsafe practices, Lower = sustainable farming",
        "Encroachments": "Higher = illegal land use, Lower = protected land",
        "IneffectiveDisasterPreparedness": "Higher = poor preparedness, Lower = ready systems",
        "DrainageSystems": "Higher = blocked drains, Lower = efficient drainage",
        "CoastalVulnerability": "Higher = closer to coast, Lower = inland",
        "Landslides": "Higher = landslide-prone area, Lower = stable land",
        "Watersheds": "Higher = poor watershed health, Lower = good water retention",
        "DeterioratingInfrastructure": "Higher = weak infrastructure, Lower = strong structures",
        "PopulationScore": "Higher = high population pressure, Lower = low density",
        "WetlandLoss": "Higher = wetlands destroyed, Lower = wetlands preserved",
        "InadequatePlanning": "Higher = poor planning, Lower = proper planning",
        "PoliticalFactors": "Higher = governance issues, Lower = effective governance"
    }

    data = {}

    for feature, description in feature_help.items():
        data[feature] = st.slider(
            f"{feature} ({description})",
            0.0,
            1.0,
            0.5
        )

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("📊 Input Values")
st.write(input_df)

if st.button("Predict Flood Risk"):

    if elevation is None:
        st.warning("📍 Please select a location on the map first.")
        st.stop()

    base_prediction = model.predict(input_df)[0]

    # Elevation adjustment
    elev_adjustment = elevation_risk_factor(elevation)

    # Flood zone adjustment (from STEP 2)
    zone_label, zone_risk = flood_zone_risk(lat, lon, elevation)

    # Combine ML + geography
    final_prediction = base_prediction + elev_adjustment + zone_risk

    # 🚨 HARD SAFETY OVERRIDE (EXTREME COASTAL LOGIC)
    if elevation is not None and elevation < 10:
        if base_prediction >= 0.20:
            final_prediction = max(final_prediction, 0.75)

    # Clamp to valid probability range
    final_prediction = max(0.0, min(1.0, final_prediction))

    # Save results
    st.session_state.final_prediction = final_prediction
    st.session_state.band_text, st.session_state.band_color = risk_band(final_prediction)
    # ===============================
    # Add marker with place + risk
    # ===============================
    risk_text, risk_color = risk_band(final_prediction)

    popup_text = f"""
    <b>Place:</b> {place_name}<br>
    <b>Flood Risk:</b> {risk_text}<br>
    <b>Probability:</b> {final_prediction:.2f}
    """

    folium.Marker(
        [lat, lon],
        popup=popup_text,
        icon=folium.Icon(color=risk_color)
    ).add_to(m)




# ===============================
# Display results (SAFE ZONE)
# ===============================
if st.session_state.band_text is not None:
    st.markdown(
        f"<h3 style='color:{st.session_state.band_color};'>"
        f"Flood Risk Level: {st.session_state.band_text}</h3>",
        unsafe_allow_html=True
    )

    st.success(
        f"🌧️ Flood Probability: {st.session_state.final_prediction:.2f}"
    )

    # ---------------------------
    # Safety Advice (Below Risk Band) 
    # ---------------------------
    if st.session_state.final_prediction < 0.30:
        risk_level = "Low"
    elif st.session_state.final_prediction < 0.55:
        risk_level = "Medium"
    else:
        risk_level = "High"

    st.subheader(f"📋 Safety Advice - {risk_level} Risk")

    advice = {
        "General": {
            "Low": "✅ Stay informed.\n✅ Keep emergency contacts handy.\n✅ Watch local weather alerts.",
            "Medium": "⚠️ Prepare emergency kit.\n⚠️ Stay ready to evacuate.\n⚠️ Avoid waterlogged areas.",
            "High": "🚨 Evacuate immediately.\n🚨 Turn off gas/electricity.\n🚨 Follow government instructions.",
        },
        "Farmer": {
            "Low": "✅ Secure storage for seeds/tools.\n✅ Monitor irrigation levels.",
            "Medium": "⚠️ Move livestock to safe zones.\n⚠️ Cover crop supplies.\n⚠️ Inspect water channels.",
            "High": "🚨 Evacuate livestock.\n🚨 Protect tractors/equipment.\n🚨 Stay in touch with agri dept.",
        },
        "Family": {
            "Low": "✅ Keep basic supplies.\n✅ Keep family informed.\n✅ Prepare child safety plan.",
            "Medium": "⚠️ Pack essentials.\n⚠️ Stay in touch with relatives.\n⚠️ Charge phones and power banks.",
            "High": "🚨 Evacuate to a safe shelter.\n🚨 Carry documents and meds.\n🚨 Avoid floodwaters.",
        },
        "Elderly": {
            "Low": "✅ Keep medications close.\n✅ Stay in touch with a neighbor.",
            "Medium": "⚠️ Have help on standby.\n⚠️ Pack health documents.\n⚠️ Limit movement outdoors.",
            "High": "🚨 Call for immediate help.\n🚨 Avoid stairs/water.\n🚨 Stay with a trusted caregiver.",
        }
    }
    # Let user choose which advice category to view
    category = st.selectbox(
        "Choose advice category:",
        ["General", "Farmer", "Family", "Elderly"]
    )

    st.markdown(advice[category][risk_level])

# ===============================
# Feature Guide / Instructions
# ===============================
feature_descriptions = {
    "MonsoonIntensity": "Amount of rainfall during monsoon season",
    "TopographyDrainage": "Quality of land drainage and slope",
    "RiverManagement": "Effectiveness of river and embankment control",
    "Deforestation": "Level of forest cover loss",
    "Urbanization": "Density of buildings and urban development",
    "ClimateChange": "Impact of climate change on the area",
    "DamsQuality": "Condition and safety of nearby dams",
    "Siltation": "Amount of silt buildup in rivers",
    "AgriculturalPractices": "Sustainability of farming methods",
    "Encroachments": "Presence of illegal land use",
    "IneffectiveDisasterPreparedness": "Level of disaster readiness",
    "DrainageSystems": "Efficiency of drainage networks",
    "CoastalVulnerability": "Proximity and exposure to coast",
    "Landslides": "Risk of landslides in the area",
    "Watersheds": "Health of local watersheds",
    "DeterioratingInfrastructure": "Condition of infrastructure",
    "PopulationScore": "Population density and pressure",
    "WetlandLoss": "Degree of wetland destruction",
    "InadequatePlanning": "Effectiveness of urban or rural planning",
    "PoliticalFactors": "Governance and policy effectiveness"
}

with st.expander("ℹ️ Feature Guide / Instructions"):
    for feature, desc in feature_descriptions.items():
        st.markdown(f"**{feature}**: {desc}")







