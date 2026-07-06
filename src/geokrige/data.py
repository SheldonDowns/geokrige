"""Core data structures: load point data, aggregate, build neighbor pairs,
and test for spatial stationarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from scipy import stats


class SpatialDataset:
    """A table of point geometries plus a numeric value column, with helpers
    for aggregation, neighbor-pair construction, and stationarity testing.

    Internally backed by a GeoDataFrame in EPSG:4326 (lon/lat degrees).
    """

    def __init__(self, gdf: gpd.GeoDataFrame, value_col: str):
        if gdf.crs is None:
            raise ValueError("gdf must have a CRS set")
        self.gdf = gdf.to_crs("EPSG:4326")
        self.value_col = value_col

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        lon_col: str,
        lat_col: str,
        value_col: str,
        crs: str = "EPSG:4326",
    ) -> "SpatialDataset":
        """Build from a plain DataFrame with separate lon/lat columns."""
        df = df.copy()
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[lon_col, lat_col, value_col])

        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        return cls(gdf, value_col)

    @classmethod
    def from_wkt(
        cls,
        df: pd.DataFrame,
        wkt_col: str,
        value_col: str,
        src_crs: str,
        dst_crs: str = "EPSG:4326",
    ) -> "SpatialDataset":
        """Build from a DataFrame with a WKT geometry column in a projected
        CRS (e.g. a state-plane system) and reproject to ``dst_crs``."""
        df = df.copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        geometry = df[wkt_col].apply(shapely_wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=src_crs)
        gdf = gdf.to_crs(dst_crs)
        gdf = gdf.dropna(subset=[value_col])
        return cls(gdf, value_col)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(
        self,
        group_cols: list[str],
        agg_funcs: tuple[str, ...] = ("mean", "std", "count"),
    ) -> "SpatialDataset":
        """Collapse repeated observations at the same location/group into
        summary statistics (e.g. mean volume per sensor per time period).

        ``group_cols`` should include whatever identifies a unique point
        (e.g. a sensor id) plus any categorical split (e.g. time period).
        Geometry is taken as the first geometry seen per group, so
        ``group_cols`` should fully determine location.
        """
        df = self.gdf.copy()
        geom_per_group = df.groupby(group_cols, observed=True)["geometry"].first()

        agg_map = {self.value_col: list(agg_funcs)}
        result = df.groupby(group_cols, observed=True).agg(agg_map)
        result.columns = [f"{self.value_col}_{f}" for f in agg_funcs]
        result = result.reset_index()

        result = result.merge(
            geom_per_group.reset_index(), on=group_cols, how="left"
        )
        gdf = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")
        new_value_col = f"{self.value_col}_mean" if "mean" in agg_funcs else f"{self.value_col}_{agg_funcs[0]}"
        return SpatialDataset(gdf, new_value_col)

    # ------------------------------------------------------------------
    # Neighbor pairs (for Moran's I and variogram construction)
    # ------------------------------------------------------------------
    def build_pairs(
        self,
        min_distance: float = 0.0,
        max_distance: float = 0.5,
        split_col: str | None = None,
    ) -> pd.DataFrame:
        """All unordered point pairs within a distance band, optionally
        only within the same category (e.g. same time period).

        Distance is computed in degrees (planar approx on EPSG:4326).
        For large datasets this is O(n^2); fine for hundreds-to-low-
        thousands of points, which covers most sensor/station networks.

        Returns a DataFrame with columns: idx_i, idx_j, val_i, val_j,
        distance, sq_diff, and ``split_col`` if given.
        """
        gdf = self.gdf.reset_index(drop=True)
        coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
        vals = gdf[self.value_col].values

        groups = gdf[split_col].values if split_col else np.zeros(len(gdf))

        rows = []
        for g in np.unique(groups):
            mask = groups == g
            idx = np.where(mask)[0]
            c = coords[idx]
            v = vals[idx]
            n = len(idx)
            for a in range(n):
                d = np.sqrt(((c[a + 1:] - c[a]) ** 2).sum(axis=1))
                keep = (d >= min_distance) & (d <= max_distance)
                for b_off in np.where(keep)[0]:
                    b = a + 1 + b_off
                    row = {
                        "idx_i": idx[a],
                        "idx_j": idx[b],
                        "val_i": v[a],
                        "val_j": v[b],
                        "distance": d[b_off],
                        "sq_diff": (v[a] - v[b]) ** 2,
                    }
                    if split_col:
                        row[split_col] = g
                    rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Stationarity
    # ------------------------------------------------------------------
    def quadrant_labels(self, lon_split: float | None = None, lat_split: float | None = None) -> pd.Series:
        """Label each point NW/NE/SW/SE by splitting on the median (or a
        given) longitude/latitude."""
        x = self.gdf.geometry.x
        y = self.gdf.geometry.y
        lon_split = x.median() if lon_split is None else lon_split
        lat_split = y.median() if lat_split is None else lat_split

        labels = np.where(
            x < lon_split,
            np.where(y > lat_split, "NW", "SW"),
            np.where(y > lat_split, "NE", "SE"),
        )
        return pd.Series(labels, index=self.gdf.index, name="quadrant")

    def test_stationarity(
        self,
        split_col: str | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Kruskal-Wallis test of whether the mean value differs across
        spatial quadrants — i.e. whether a global trend (non-stationarity)
        is present. If non-stationary, prefer Universal over Ordinary
        Kriging.

        If ``split_col`` is given (e.g. a time-period column), the test is
        run separately within each group.

        Returns a DataFrame with one row per group: H statistic, p-value,
        and a ``stationary`` boolean.
        """
        df = self.gdf.copy()
        df["quadrant"] = self.quadrant_labels()

        groups_iter = df.groupby(split_col) if split_col else [(None, df)]
        results = []
        for key, sub in groups_iter:
            samples = [
                g[self.value_col].values
                for _, g in sub.groupby("quadrant")
                if len(g) > 0
            ]
            samples = [s for s in samples if len(s) > 0]
            if len(samples) < 2:
                continue
            h, p = stats.kruskal(*samples)
            results.append(
                {
                    split_col: key,
                    "H_statistic": h,
                    "p_value": p,
                    "stationary": p >= alpha,
                }
                if split_col
                else {"H_statistic": h, "p_value": p, "stationary": p >= alpha}
            )
        return pd.DataFrame(results)
