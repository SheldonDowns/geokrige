# Quickstart

## Install

```bash
pip install geokrige
pip install "geokrige[export]"   # adds folium + rasterio for exports
```

## Basic pipeline

```python
from geokrige import SpatialDataset, Variogram, KrigingPipeline, morans_i, export

# 1. Load points (lon/lat + a value column)
ds = SpatialDataset.from_dataframe(df, lon_col="lon", lat_col="lat", value_col="volume")

# 2. Optionally collapse repeated observations per location/category
ds = ds.aggregate(group_cols=["sensor_id", "time_period"])

# 3. Check spatial autocorrelation
pairs = ds.build_pairs(max_distance=0.02)
print(morans_i(pairs))

# 4. Fit a variogram
vario = Variogram(pairs, bin_size=0.01).fit(model="auto")

# 5. Krige (auto-picks Ordinary vs Universal)
pipeline = KrigingPipeline(ds, vario, kriging_type="auto")
grid_lon, grid_lat, z, variance = pipeline.predict_grid(
    bounds=(-74.26, -73.70, 40.49, 40.92), resolution=80
)

# 6. Validate
cv = pipeline.cross_validate(k=5)
print("RMSE:", cv["rmse"])

# 7. Export
export.to_heatmap(grid_lon, grid_lat, z, ds, "surface.html")
export.to_raster(z, bounds=(-74.26, 40.49, -73.70, 40.92), out_path="surface.tif")
```

## Fetching from a Socrata portal

```python
from geokrige import fetch_socrata

df = fetch_socrata(
    domain="data.cityofnewyork.us",
    dataset_id="7ym2-wayt",
    app_token="YOUR_TOKEN",
)
```

## Choosing Ordinary vs Universal Kriging manually

By default `KrigingPipeline(..., kriging_type="auto")` runs a Kruskal-Wallis
test across spatial quadrants and picks Universal Kriging (with a linear
drift) if it detects a significant trend, otherwise Ordinary Kriging. You
can override this:

```python
pipeline = KrigingPipeline(ds, vario, kriging_type="ordinary")
```
