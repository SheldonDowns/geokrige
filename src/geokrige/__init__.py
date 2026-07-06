"""
geokrige
========

A toolkit for spatial point-data analysis and interpolation:
fetch -> aggregate -> test spatial autocorrelation (Moran's I) ->
fit a variogram -> krige (Ordinary/Universal, auto-selected) ->
cross-validate -> export (heatmap / raster / shapefile).

Designed to be dataset-agnostic: works for traffic sensors, air quality
monitors, real-estate listings, crime incidents, or any point dataset
with a longitude, latitude, and numeric value.
"""

from .data import SpatialDataset
from .autocorrelation import morans_i
from .variogram import Variogram
from .kriging import KrigingPipeline
from .socrata import fetch_socrata
from . import export

__version__ = "0.1.0"

__all__ = [
    "SpatialDataset",
    "morans_i",
    "Variogram",
    "KrigingPipeline",
    "fetch_socrata",
    "export",
]
