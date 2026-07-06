"""Global Moran's I: tests whether nearby points have similar values
(positive spatial autocorrelation), a prerequisite for Kriging to be
meaningful — if values are spatially random, interpolation is unjustified."""

from __future__ import annotations

import pandas as pd


def morans_i(pairs: pd.DataFrame) -> dict:
    """Compute inverse-distance-weighted global Moran's I from a pairs
    DataFrame (as produced by ``SpatialDataset.build_pairs``).

    Pairs must have columns: val_i, val_j, distance, and only pairs with
    distance > 0 should be passed in (self-pairs / duplicate locations
    excluded).

    Returns dict with n_pairs, mean_val, and morans_i. Values roughly in
    [-1, 1]; positive => similar values cluster together (the prerequisite
    for Kriging to be meaningful).
    """
    pairs = pairs[pairs["distance"] > 0]
    if len(pairs) == 0:
        return {"n_pairs": 0, "mean_val": float("nan"), "morans_i": float("nan")}

    mean_val = pairs["val_i"].mean()
    w = 1.0 / pairs["distance"]

    numerator = (w * (pairs["val_i"] - mean_val) * (pairs["val_j"] - mean_val)).sum()
    denominator = (w.sum() * ((pairs["val_i"] - mean_val) ** 2).sum()) / len(pairs)

    moran = numerator / denominator if denominator != 0 else float("nan")

    return {
        "n_pairs": len(pairs),
        "mean_val": mean_val,
        "morans_i": moran,
    }


def morans_i_by_group(pairs: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Run ``morans_i`` separately for each value of ``group_col`` (e.g.
    time period)."""
    rows = []
    for key, sub in pairs.groupby(group_col):
        result = morans_i(sub)
        result[group_col] = key
        rows.append(result)
    return pd.DataFrame(rows)
