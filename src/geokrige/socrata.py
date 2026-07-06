"""Generic fetcher for Socrata-powered open data portals (data.cityofnewyork.us,
data.cdc.gov, data.lacity.org, etc.) — not tied to any one city or dataset."""

from __future__ import annotations

import pandas as pd
import requests


def fetch_socrata(
    domain: str,
    dataset_id: str,
    app_token: str | None = None,
    limit: int = 50_000,
    where: str | None = None,
    select: str | None = None,
    extra_params: dict | None = None,
) -> pd.DataFrame:
    """Fetch a dataset from any Socrata Open Data API (SODA) portal.

    Parameters
    ----------
    domain : e.g. "data.cityofnewyork.us"
    dataset_id : the 4x4 dataset identifier, e.g. "7ym2-wayt"
    app_token : optional Socrata app token (raises rate limits)
    limit : max rows to pull
    where : optional SoQL $where clause
    select : optional SoQL $select clause
    extra_params : any additional SoQL query params ($group, $order, etc.)

    Returns
    -------
    pandas.DataFrame
    """
    url = f"https://{domain}/resource/{dataset_id}.json"
    headers = {"X-App-Token": app_token} if app_token else {}

    params = {"$limit": limit}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if extra_params:
        params.update(extra_params)

    response = requests.get(url, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json())
