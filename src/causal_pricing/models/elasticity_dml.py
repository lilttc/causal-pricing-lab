from __future__ import annotations

from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from doubleml import DoubleMLData, DoubleMLPLR


def estimate_dml_elasticity(
    df: pd.DataFrame,
    X_controls: pd.DataFrame,
    product: str,
    qty_col: str = "qty",
    price_col: str = "avg_price",
    min_obs: int = 30,
    n_folds: int = 5,
    random_state: int = 42,
) -> Optional[Dict]:
    """
    Estimate own-price elasticity for a single product using Double Machine Learning (DML).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset at daily × product × city grain.
    X_controls : pd.DataFrame
        Controls matrix aligned to df.index (time dummies, city FE, etc.).
    product : str
        Product name to filter.
    qty_col : str, default="qty"
        Quantity column.
    price_col : str, default="avg_price"
        Price column.
    min_obs : int, default=30
        Minimum observations required.
    n_folds : int, default=5
        Number of folds for cross-fitting.
    random_state : int, default=42
        Random seed for learners.

    Returns
    -------
    dict or None
        Dict with product, coef (level effect), elasticity (log-approx), nobs.
        None if insufficient variation.
    """
    df_p = df[df["product"] == product].copy()
    if df_p[price_col].nunique() < 2 or len(df_p) < min_obs:
        return None

    Y = df_p[qty_col].values
    T = df_p[price_col].values.reshape(-1, 1)
    X = X_controls.loc[df_p.index].values

    ml_g = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state)
    ml_m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state)

    dml_data = DoubleMLData.from_arrays(X, Y, T)
    dml_plr = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=n_folds)
    dml_plr.fit()

    coef = dml_plr.coef[0]
    mean_q = df_p[qty_col].mean()
    mean_p = df_p[price_col].mean()
    elasticity = coef * (mean_p / mean_q)

    return dict(
        product=product,
        coef=float(coef),
        elasticity=float(elasticity),
        nobs=len(df_p),
    )


def batch_dml_elasticities(
    df: pd.DataFrame,
    X_controls: pd.DataFrame,
    products: List[str],
    **kwargs,
) -> pd.DataFrame:
    """
    Estimate elasticities for a list of products.

    Returns
    -------
    pd.DataFrame
        One row per product with columns [product, coef, elasticity, nobs].
    """
    results = []
    for p in products:
        out = estimate_dml_elasticity(df, X_controls, p, **kwargs)
        if out:
            results.append(out)
    return pd.DataFrame(results)
