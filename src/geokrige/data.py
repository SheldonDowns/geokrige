"""Core data structures, backed by Apache SedonaDB (sedona.db) rather than
GeoPandas. All geometry construction, transforms, aggregation, and neighbor-
pair building are pushed down into SedonaDB SQL, which uses Arrow/DataFusion
under the hood and scales to far larger point sets than an in-memory
GeoPandas + Python-loop approach.

Rows leaving SedonaDB (into pandas, for scipy/pykrige/sklearn, none of which
understand SedonaDB DataFrames) are pulled out only at the last moment, via
``.to_pandas()`` / ``.to_geopandas()``.
"""

from __future__ import annotations

import itertools
import uuid

import pandas as pd
from scipy import stats

import sedona.db

_counter = itertools.count()


def _tmp_view_name(prefix: str) -> str:
    return f"_geokrige_{prefix}_{next(_counter)}_{uuid.uuid4().hex[:6]}"


class SpatialDataset:
    """A SedonaDB-backed table of point geometries plus a numeric value
    column, with helpers for aggregation, neighbor-pair construction (as a
    spatial self-join), and stationarity testing.

    Internally, geometries are stored in EPSG:4326 (lon/lat degrees) in a
    SedonaDB view. Use ``.sd.sql(...)`` directly for any custom SQL you need
    beyond the provided helpers.
    """

    def __init__(self, sd, view_name: str, value_col: str):
        self.sd = sd
        self.view_name = view_name
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
        sd=None,
    ) -> "SpatialDataset":
        """Build from a plain DataFrame with separate lon/lat columns
        (assumed already EPSG:4326)."""
        sd = sd or sedona.db.connect()
        raw_view = _tmp_view_name("raw")
        out_view = _tmp_view_name("pts")

        sd.create_data_frame(df).to_view(raw_view, overwrite=True)
        sd.sql(f"""
            SELECT *, ST_Point(
                CAST({lon_col} AS DOUBLE), CAST({lat_col} AS DOUBLE), 4326
            ) AS geometry
            FROM {raw_view}
            WHERE {lon_col} IS NOT NULL
              AND {lat_col} IS NOT NULL
              AND {value_col} IS NOT NULL
        """).to_view(out_view, overwrite=True)

        return cls(sd, out_view, value_col)

    @classmethod
    def from_wkt(
        cls,
        df: pd.DataFrame,
        wkt_col: str,
        value_col: str,
        src_crs: str,
        dst_crs: str = "EPSG:4326",
        sd=None,
    ) -> "SpatialDataset":
        """Build from a DataFrame with a WKT geometry column in a projected
        CRS (e.g. a state-plane system) and reproject to ``dst_crs``."""
        sd = sd or sedona.db.connect()
        raw_view = _tmp_view_name("raw")
        out_view = _tmp_view_name("pts")

        sd.create_data_frame(df).to_view(raw_view, overwrite=True)
        sd.sql(f"""
            SELECT *, ST_Transform(
                ST_GeomFromWKT({wkt_col}), '{src_crs}', '{dst_crs}'
            ) AS geometry
            FROM {raw_view}
            WHERE {wkt_col} IS NOT NULL AND {value_col} IS NOT NULL
        """).to_view(out_view, overwrite=True)

        return cls(sd, out_view, value_col)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def lon_lat_val(self):
        """Pull lon, lat, and value as three numpy arrays — the format
        pykrige/scipy/sklearn need, since none of them understand SedonaDB
        DataFrames directly."""
        df = self.sd.sql(f"""
            SELECT ST_X(geometry) AS _lon, ST_Y(geometry) AS _lat, {self.value_col} AS _val
            FROM {self.view_name}
        """).to_pandas()
        return df["_lon"].values, df["_lat"].values, df["_val"].values

    def to_pandas(self) -> pd.DataFrame:
        """Pull the full table out of SedonaDB as a plain pandas DataFrame
        (geometry column comes back as WKB bytes)."""
        return self.sd.sql(f"SELECT * FROM {self.view_name}").to_pandas()

    def to_geopandas(self):
        """Pull the full table out as a GeoDataFrame, e.g. for plotting or
        interop with other GeoPandas-based tools. Requires the optional
        ``geopandas`` dependency (``pip install geokrige[export]``).

        SedonaDB's ``to_pandas()`` already returns geometry as a GeoPandas
        GeometryArray with its own CRS (``OGC:CRS84``, equivalent to
        EPSG:4326) attached, so this just wraps it and normalizes the CRS
        label rather than re-declaring a conflicting one.
        """
        import geopandas as gpd

        df = self.to_pandas()
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        return gdf.to_crs("EPSG:4326")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(
        self,
        group_cols: list[str],
        agg_funcs: tuple[str, ...] = ("mean", "std", "count"),
    ) -> "SpatialDataset":
        """Collapse repeated observations at the same location/group into
        summary statistics (e.g. mean volume per sensor per time period),
        pushed down as a SedonaDB ``GROUP BY``.

        ``group_cols`` should fully determine location (geometry is included
        in the GROUP BY alongside them, matching the one-geometry-per-group
        assumption).
        """
        sql_funcs = {"mean": "AVG", "std": "STDDEV", "count": "COUNT", "sum": "SUM", "min": "MIN", "max": "MAX"}
        group_list = ", ".join(group_cols + ["geometry"])

        select_parts_sql = ", ".join(
            f"{sql_funcs[f]}({self.value_col}) AS {self.value_col}_{f}"
            if f != "count" else f"COUNT(*) AS {self.value_col}_count"
            for f in agg_funcs
        )

        out_view = _tmp_view_name("agg")
        self.sd.sql(f"""
            SELECT {', '.join(group_cols)}, geometry, {select_parts_sql}
            FROM {self.view_name}
            GROUP BY {group_list}
        """).to_view(out_view, overwrite=True)

        new_value_col = f"{self.value_col}_mean" if "mean" in agg_funcs else f"{self.value_col}_{agg_funcs[0]}"
        return SpatialDataset(self.sd, out_view, new_value_col)

    # ------------------------------------------------------------------
    # Neighbor pairs (for Moran's I and variogram construction)
    # ------------------------------------------------------------------
    def build_pairs(
        self,
        min_distance: float = 0.0,
        max_distance: float = 0.5,
        split_col: str | None = None,
        id_col: str | None = None,
    ) -> pd.DataFrame:
        """All unordered point pairs within a distance band, optionally
        only within the same category (e.g. same time period), computed as
        a SedonaDB spatial self-join.

        Distance is computed in degrees (planar approx on EPSG:4326).
        Pushing this down into SQL (vs. a Python double loop) is what lets
        this scale past a few thousand points.

        Returns a DataFrame with columns: idx_i, idx_j, val_i, val_j,
        distance, sq_diff, and ``split_col`` if given.
        """
        id_col = id_col or "_geokrige_row_id"
        base_view = _tmp_view_name("withid")
        self.sd.sql(f"""
            SELECT *, ROW_NUMBER() OVER () AS {id_col}
            FROM {self.view_name}
        """).to_view(base_view, overwrite=True)

        split_join = f"AND a.{split_col} = b.{split_col}" if split_col else ""
        split_select = f", a.{split_col} AS {split_col}" if split_col else ""

        query = f"""
            SELECT
                a.{id_col} AS idx_i,
                b.{id_col} AS idx_j,
                a.{self.value_col} AS val_i,
                b.{self.value_col} AS val_j,
                ST_Distance(a.geometry, b.geometry) AS distance,
                POWER(a.{self.value_col} - b.{self.value_col}, 2) AS sq_diff
                {split_select}
            FROM {base_view} a
            JOIN {base_view} b
              ON a.{id_col} < b.{id_col}
              {split_join}
            WHERE ST_Distance(a.geometry, b.geometry) BETWEEN {min_distance} AND {max_distance}
        """
        return self.sd.sql(query).to_pandas()

    # ------------------------------------------------------------------
    # Stationarity
    # ------------------------------------------------------------------
    def test_stationarity(
        self,
        split_col: str | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Kruskal-Wallis test of whether the mean value differs across
        spatial quadrants (NW/NE/SW/SE, split on median lon/lat) — i.e.
        whether a global trend (non-stationarity) is present. If
        non-stationary, prefer Universal over Ordinary Kriging.

        Quadrant assignment runs as SedonaDB SQL; the Kruskal-Wallis test
        itself uses scipy, since SedonaDB has no built-in equivalent.

        If ``split_col`` is given (e.g. a time-period column), the test is
        run separately within each group.

        Returns a DataFrame with one row per group: H statistic, p-value,
        and a ``stationary`` boolean.
        """
        labeled_view = _tmp_view_name("quad")
        self.sd.sql(f"""
            WITH medians AS (
                SELECT
                    approx_percentile_cont(ST_X(geometry), 0.5) AS lon_med,
                    approx_percentile_cont(ST_Y(geometry), 0.5) AS lat_med
                FROM {self.view_name}
            )
            SELECT t.*,
                CASE
                    WHEN ST_X(t.geometry) < m.lon_med AND ST_Y(t.geometry) > m.lat_med THEN 'NW'
                    WHEN ST_X(t.geometry) < m.lon_med AND ST_Y(t.geometry) <= m.lat_med THEN 'SW'
                    WHEN ST_X(t.geometry) >= m.lon_med AND ST_Y(t.geometry) > m.lat_med THEN 'NE'
                    ELSE 'SE'
                END AS quadrant
            FROM {self.view_name} t
            CROSS JOIN medians m
        """).to_view(labeled_view, overwrite=True)

        df = self.sd.sql(f"SELECT * FROM {labeled_view}").to_pandas()

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
            row = {"H_statistic": h, "p_value": p, "stationary": p >= alpha}
            if split_col:
                row = {split_col: key, **row}
            results.append(row)
        return pd.DataFrame(results)
