from geo_features.dem_utils import get_elevation
from geo_features.features import build_features
import joblib

# Load trained model
model = joblib.load("models/flood_model.pkl")

# Test location (Odisha)
lat, lon = 20.5, 84.0

# Sample inputs
rainfall = 180      # mm
river_level = 5.2   # meters

# Get elevation from DEM
elevation = get_elevation(lat, lon)

# Build feature vector
X = build_features(rainfall, river_level, elevation)

# Predict
prediction = model.predict(X)

print("Elevation:", elevation)
print("Flood Risk Prediction:", prediction)
