"""Kriging pipeline built on PyKrige: chooses Ordinary vs Universal Kriging
based on a stationarity test, predicts over a grid, and supports k-fold
cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from .data import SpatialDataset
from .variogram import Variogram


class KrigingPipeline:
    """Fit and apply Kriging interpolation over a SpatialDataset.

    Parameters
    ----------
    dataset : a SpatialDataset of points to interpolate from
    variogram : a fitted Variogram instance
    kriging_type : "auto", "ordinary", or "universal".
        "auto" runs a stationarity test (Kruskal-Wallis across quadrants)
        and picks Universal Kriging with a linear drift if a significant
        spatial trend is detected, otherwise Ordinary Kriging.
    drift_terms : drift terms for Universal Kriging (default: regional_linear)
    """

    def __init__(
        self,
        dataset: SpatialDataset,
        variogram: Variogram,
        kriging_type: str = "auto",
        drift_terms: list[str] | None = None,
    ):
        self.dataset = dataset
        self.variogram = variogram
        self.drift_terms = drift_terms or ["regional_linear"]

        if kriging_type == "auto":
            stat = dataset.test_stationarity()
            is_stationary = bool(stat["stationary"].iloc[0]) if len(stat) else True
            kriging_type = "ordinary" if is_stationary else "universal"
        self.kriging_type = kriging_type

    def _build_model(self, lon, lat, val):
        params = self.variogram.pykrige_params()
        model_name = self.variogram.model_name
        if self.kriging_type == "ordinary":
            return OrdinaryKriging(
                lon, lat, val,
                variogram_model=model_name,
                variogram_parameters=params,
                verbose=False,
            )
        return UniversalKriging(
            lon, lat, val,
            variogram_model=model_name,
            variogram_parameters=params,
            drift_terms=self.drift_terms,
            verbose=False,
        )

    def predict_grid(self, bounds: tuple[float, float, float, float], resolution: int = 80):
        """Predict over a regular grid.

        bounds : (lon_min, lon_max, lat_min, lat_max)
        resolution : number of points per axis

        Returns (grid_lon, grid_lat, z, variance)
        """
        lon_min, lon_max, lat_min, lat_max = bounds
        lon, lat, val = self.dataset.lon_lat_val()

        model = self._build_model(lon, lat, val)
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        z, ss = model.execute("grid", grid_lon, grid_lat)
        return grid_lon, grid_lat, z, ss

    def cross_validate(self, k: int = 5, random_state: int = 42) -> dict:
        """K-fold cross validation. Returns RMSE plus the raw actual/
        predicted arrays for residual plotting."""
        lon, lat, val = self.dataset.lon_lat_val()

        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        actuals, predicted = [], []

        for train_idx, test_idx in kf.split(lon):
            model = self._build_model(lon[train_idx], lat[train_idx], val[train_idx])
            z_pred, _ = model.execute("points", lon[test_idx], lat[test_idx])
            actuals.extend(val[test_idx])
            predicted.extend(z_pred)

        actuals = np.array(actuals)
        predicted = np.array(predicted)
        rmse = float(np.sqrt(mean_squared_error(actuals, predicted)))

        return {
            "rmse": rmse,
            "actuals": actuals,
            "predicted": predicted,
            "residuals": actuals - predicted,
        }
