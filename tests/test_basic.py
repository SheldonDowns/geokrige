"""Basic smoke tests using a synthetic spatially-correlated dataset, so the
suite runs without network access or real sensor data."""

import numpy as np
import pandas as pd
import pytest

from geokrige import SpatialDataset, Variogram, KrigingPipeline, morans_i


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 80
    lon = rng.uniform(-74.2, -73.8, n)
    lat = rng.uniform(40.5, 40.9, n)
    # value with a spatial trend + noise, so it's non-stationary and
    # spatially autocorrelated -- a realistic test case
    value = 100 * (lon + 74.2) + 50 * (lat - 40.5) + rng.normal(0, 5, n)
    return pd.DataFrame({"lon": lon, "lat": lat, "value": value})


def test_spatial_dataset_from_dataframe(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    assert len(ds.gdf) == len(synthetic_df)
    assert ds.value_col == "value"


def test_build_pairs(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    pairs = ds.build_pairs(max_distance=1.0)
    n = len(synthetic_df)
    assert len(pairs) == n * (n - 1) // 2


def test_morans_i_positive_for_trend(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    pairs = ds.build_pairs(max_distance=1.0)
    result = morans_i(pairs)
    assert result["morans_i"] > 0  # trending data should show positive autocorrelation


def test_stationarity_detects_trend(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    stat = ds.test_stationarity()
    assert stat["stationary"].iloc[0] == False  # noqa: E712 -- strong trend by construction


def test_variogram_fit(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    pairs = ds.build_pairs(max_distance=1.0)
    vario = Variogram(pairs, bin_size=0.05).fit(model="auto")
    assert vario.model_name in {"spherical", "exponential", "gaussian"}
    assert vario.params["range"] > 0


def test_kriging_pipeline_grid_predict(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    pairs = ds.build_pairs(max_distance=1.0)
    vario = Variogram(pairs, bin_size=0.05).fit(model="auto")
    pipeline = KrigingPipeline(ds, vario, kriging_type="auto")
    assert pipeline.kriging_type == "universal"  # trend present

    grid_lon, grid_lat, z, ss = pipeline.predict_grid(
        bounds=(-74.2, -73.8, 40.5, 40.9), resolution=10
    )
    assert z.shape == (10, 10)
    assert ss.shape == (10, 10)


def test_cross_validate(synthetic_df):
    ds = SpatialDataset.from_dataframe(synthetic_df, "lon", "lat", "value")
    pairs = ds.build_pairs(max_distance=1.0)
    vario = Variogram(pairs, bin_size=0.05).fit(model="auto")
    pipeline = KrigingPipeline(ds, vario, kriging_type="ordinary")
    cv = pipeline.cross_validate(k=4)
    assert cv["rmse"] >= 0
    assert len(cv["actuals"]) == len(synthetic_df)
