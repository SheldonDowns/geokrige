"""Export Kriging output to interactive maps, GeoTIFF rasters, and
shapefiles."""

from __future__ import annotations

import numpy as np

from .data import SpatialDataset


def to_heatmap(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    z: np.ndarray,
    dataset: SpatialDataset,
    out_path: str,
    title: str | None = None,
    tiles: str = "CartoDB positron",
):
    """Save an interactive Folium heatmap of the prediction surface, with
    actual sample points overlaid. Requires the optional ``folium`` dep."""
    import folium
    from folium.plugins import HeatMap

    heat_data = []
    for i, lat in enumerate(grid_lat):
        for j, lon in enumerate(grid_lon):
            val = float(z[i, j])
            if val > 0:
                heat_data.append([lat, lon, val])

    center_lat = float(np.mean(grid_lat))
    center_lon = float(np.mean(grid_lon))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=tiles)

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_val=float(z.max()),
        radius=20,
        blur=25,
        gradient={"0.2": "blue", "0.4": "cyan", "0.6": "yellow", "0.8": "orange", "1.0": "red"},
    ).add_to(m)

    for _, row in dataset.to_geopandas().iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color="black",
            fill=True,
            fill_opacity=0.5,
            popup=f"Value: {row[dataset.value_col]:.1f}",
        ).add_to(m)

    if title:
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
             background-color: white; padding: 10px; border-radius: 5px;
             box-shadow: 2px 2px 5px rgba(0,0,0,0.3); font-family: Arial;">
            <b>{title}</b>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

    m.save(out_path)
    return m


def to_raster(
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    out_path: str,
    crs: str = "EPSG:4326",
):
    """Save a 2D array as a GeoTIFF. bounds = (lon_min, lat_min, lon_max, lat_max).
    Requires the optional ``rasterio`` dep."""
    import rasterio
    from rasterio.transform import from_bounds

    lon_min, lat_min, lon_max, lat_max = bounds
    nrows, ncols = z.shape
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)

    data = np.flipud(np.asarray(z))
    with rasterio.open(
        out_path, "w", driver="GTiff", height=nrows, width=ncols,
        count=1, dtype=data.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)


def to_shapefile(dataset: SpatialDataset, out_path: str):
    """Save the dataset's points (e.g. sensor locations) as a shapefile."""
    dataset.to_geopandas().to_file(out_path)
