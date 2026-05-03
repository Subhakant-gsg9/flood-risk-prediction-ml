import rasterio

DEM_PATH = "geo_features/rasters_SRTMGL1/output_SRTMGL1.tif"

with rasterio.open(DEM_PATH) as dem:
    DEM_DATA = dem.read(1)
    TRANSFORM = dem.transform
    HEIGHT, WIDTH = DEM_DATA.shape

def get_elevation(lat, lon):
    """
    Returns elevation (meters) for given latitude & longitude
    """
    # IMPORTANT: rasterio expects (lon, lat)
    col, row = ~TRANSFORM * (lon, lat)

    row, col = int(row), int(col)

    # Bounds check (prevents crashes)
    if 0 <= row < HEIGHT and 0 <= col < WIDTH:
        return float(DEM_DATA[row, col])
    else:
        return None
