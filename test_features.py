from geo_features.features import build_features

X = build_features(
    rainfall=180,
    river_level=5.2,
    elevation=399
)

print(X)
