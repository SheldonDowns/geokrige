# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-06

### Added
- `SpatialDataset`: load points from a DataFrame (lon/lat or WKT), aggregate
  repeated observations, build neighbor pairs, test spatial stationarity.
- `morans_i` / `morans_i_by_group`: global Moran's I spatial autocorrelation.
- `Variogram`: empirical variogram binning with spherical/exponential/gaussian
  model fitting (auto model selection).
- `KrigingPipeline`: Ordinary/Universal Kriging with automatic type selection,
  grid prediction, and k-fold cross-validation.
- `export`: Folium heatmap, GeoTIFF raster, and shapefile export.
- `fetch_socrata`: generic fetcher for any Socrata (SODA) open-data portal.
