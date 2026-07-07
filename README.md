# geokrige

Fetch, test, and krige spatial point data — a small pipeline for turning
scattered sensor/station/listing observations into a smooth interpolated
surface, with the diagnostics to justify it along the way:

`fetch → aggregate → Moran's I → variogram fit → Kriging (auto OK/UK) → cross-validate → export`

Works with any point dataset that has a longitude, latitude, and a numeric
value — traffic counters, air quality monitors, real-estate prices, crime
incidents, soil samples, etc.

Geometry construction, transforms, aggregation, and neighbor-pair building
are all pushed down into [Apache SedonaDB](https://sedona.apache.org/sedonadb/latest/)
(`sedona.db`) SQL rather than GeoPandas — a single-node, Arrow/DataFusion-backed
spatial engine, so the same pipeline scales from a few dozen points to
datasets far larger than fit comfortably in a GeoPandas + Python-loop
approach, with no Spark cluster required.

## Install

Not yet published to PyPI — install directly from GitHub:

```bash
pip install "geokrige @ git+https://github.com/SheldonDowns/geokrige.git"
# optional export targets (interactive maps, GeoTIFF):
pip install "geokrige[export] @ git+https://github.com/SheldonDowns/geokrige.git"
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add "geokrige @ git+https://github.com/SheldonDowns/geokrige.git"
uv add "geokrige[export] @ git+https://github.com/SheldonDowns/geokrige.git"
```

<!--
Once published to PyPI, switch to:
    pip install geokrige
    pip install "geokrige[export]"
-->

## Quickstart

```python
from geokrige import SpatialDataset, Variogram, KrigingPipeline, morans_i, export

# 1. Load points (lon/lat + a value column)
ds = SpatialDataset.from_dataframe(df, lon_col="lon", lat_col="lat", value_col="volume")

# 2. Optionally collapse repeated observations per location/category
ds = ds.aggregate(group_cols=["sensor_id", "time_period"])

# 3. Check spatial autocorrelation — is interpolation even justified?
pairs = ds.build_pairs(max_distance=0.02)
print(morans_i(pairs))   # positive => nearby points are similar, good for Kriging

# 4. Fit a variogram (auto-selects spherical/exponential/gaussian)
vario = Variogram(pairs, bin_size=0.01).fit(model="auto")

# 5. Krige — auto-picks Ordinary vs Universal based on a stationarity test
pipeline = KrigingPipeline(ds, vario, kriging_type="auto")
grid_lon, grid_lat, z, variance = pipeline.predict_grid(
    bounds=(-74.26, -73.70, 40.49, 40.92), resolution=80
)

# 6. Validate
cv = pipeline.cross_validate(k=5)
print("RMSE:", cv["rmse"])

# 7. Export
export.to_heatmap(grid_lon, grid_lat, z, ds, "surface.html", title="Predicted Surface")
export.to_raster(z, bounds=(-74.26, 40.49, -73.70, 40.92), out_path="surface.tif")
```

## Fetching from open-data portals

```python
from geokrige import fetch_socrata

df = fetch_socrata(
    domain="data.cityofnewyork.us",
    dataset_id="7ym2-wayt",
    app_token="YOUR_TOKEN",
    limit=50_000,
)
```

## Module map

| Module               | Purpose                                                           |
|-----------------------|--------------------------------------------------------------------|
| `socrata.py`          | Pull data from any Socrata (SODA) open-data API                   |
| `data.py`             | `SpatialDataset`: SedonaDB-backed load, aggregate, build pairs, test stationarity |
| `autocorrelation.py`  | Global Moran's I                                                   |
| `variogram.py`        | Empirical variogram + spherical/exponential/gaussian fit           |
| `kriging.py`          | `KrigingPipeline`: auto OK/UK, grid predict, cross-validate         |
| `export.py`           | Folium heatmap, GeoTIFF raster, shapefile export (needs `geokrige[export]`) |

`SpatialDataset` wraps a SedonaDB view rather than an in-memory GeoDataFrame.
Use `ds.sd.sql("...")` for any custom SQL, `ds.to_pandas()` for a plain
DataFrame (geometry comes back as shapely `Point` objects), or
`ds.to_geopandas()` (requires the `export` extra) for GeoPandas interop.

## Development

```bash
git clone https://github.com/yourusername/geokrige
cd geokrige
uv sync --all-extras
uv run pytest
```

See `CONTRIBUTING.md` for guidelines and `CHANGELOG.md` for release history.

## Docs

Full documentation: https://geokrige.readthedocs.io

## License

MIT — see `LICENSE`.
