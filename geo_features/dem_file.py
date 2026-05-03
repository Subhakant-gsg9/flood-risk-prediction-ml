import rasterio

# Step 1: Open DEM file
dem_path = "geo_features/rasters_SRTMGL1/output_SRTMGL1.tif"


with rasterio.open(dem_path) as dem:
    print("✅ DEM loaded successfully!")
    print("DEM Bounds:", dem.bounds)
    # Step 2: Coordinates (Longitude, Latitude)
    lon, lat = 84.0, 20.5  # You can change this to any point within the DEM bounds Longitude: between 81.0 and 87.5|Latitude: between 18.0 and 22.7

    # Step 3: Convert (lon, lat) to row/col
    row, col = dem.index(lon, lat)
    print(f"Pixel coordinates for ({lon}, {lat}) ➝ row: {row}, col: {col}")

    # Step 4: Read elevation value from the raster
    elevation = dem.read(1)[row, col]
    print(f"🌄 Elevation at ({lon}, {lat}) is: {elevation} meters")
