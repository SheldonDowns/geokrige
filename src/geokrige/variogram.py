"""Empirical variogram construction and theoretical model fitting
(spherical / exponential / gaussian), with automatic best-model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _spherical(h, nugget, sill, range_):
    return np.where(
        h <= range_,
        nugget + sill * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        nugget + sill,
    )


def _exponential(h, nugget, sill, range_):
    return nugget + sill * (1 - np.exp(-h / range_))


def _gaussian(h, nugget, sill, range_):
    return nugget + sill * (1 - np.exp(-(h ** 2) / (range_ ** 2)))


MODELS = {
    "spherical": _spherical,
    "exponential": _exponential,
    "gaussian": _gaussian,
}


class Variogram:
    """Empirical variogram built from point pairs, with theoretical-model
    fitting. Use ``.empirical_df`` for plotting and ``.params`` /
    ``.model_name`` to feed into Kriging.
    """

    def __init__(self, pairs: pd.DataFrame, bin_size: float = 0.01, max_lag: float | None = None):
        self.pairs = pairs[pairs["distance"] > 0].copy()
        self.bin_size = bin_size
        self.max_lag = max_lag

        self.empirical_df = self._empirical()
        self.model_name: str | None = None
        self.params: dict | None = None

    def _empirical(self) -> pd.DataFrame:
        df = self.pairs.copy()
        if self.max_lag is not None:
            df = df[df["distance"] <= self.max_lag]
        df["lag_bin"] = np.round(df["distance"] / self.bin_size) * self.bin_size
        grouped = df.groupby("lag_bin").agg(
            num_pairs=("sq_diff", "count"),
            semivariance=("sq_diff", lambda s: s.mean() / 2),
        )
        return grouped.reset_index().sort_values("lag_bin")

    def fit(self, model: str = "auto", p0: tuple = (5000, 40000, 0.2)) -> "Variogram":
        """Fit one theoretical model, or try all three and keep the best
        (lowest sum-squared-residual) if ``model="auto"``.
        """
        h = self.empirical_df["lag_bin"].values
        gamma = self.empirical_df["semivariance"].values

        candidates = MODELS.keys() if model == "auto" else [model]
        best = None
        for name in candidates:
            fn = MODELS[name]
            try:
                popt, _ = curve_fit(fn, h, gamma, p0=p0, maxfev=10000)
                resid = np.sum((fn(h, *popt) - gamma) ** 2)
                if best is None or resid < best[2]:
                    best = (name, popt, resid)
            except RuntimeError:
                continue

        if best is None:
            raise RuntimeError("No variogram model could be fit to the empirical data")

        self.model_name, popt, _ = best
        self.params = {"nugget": popt[0], "sill": popt[1], "range": popt[2]}
        return self

    def predict(self, h: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Call .fit() before .predict()")
        fn = MODELS[self.model_name]
        return fn(h, self.params["nugget"], self.params["sill"], self.params["range"])

    def pykrige_params(self) -> dict:
        """Params formatted for pykrige's ``variogram_parameters=`` kwarg
        (it calls the sill parameter "psill")."""
        if self.params is None:
            raise RuntimeError("Call .fit() before .pykrige_params()")
        return {
            "nugget": self.params["nugget"],
            "psill": self.params["sill"],
            "range": self.params["range"],
        }
