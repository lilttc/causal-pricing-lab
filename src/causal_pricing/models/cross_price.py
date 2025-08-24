from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import statsmodels.api as sm


@dataclass(frozen=True)
class CrossPriceResult:
    """
    Container for cross-price elasticities of a focal product against a set of products.
    """
    focal: str
    coefs: pd.Series           # index: ["own"] + rivals; values: elasticities
    r2: float
    nobs: int


def _safe_log(s: pd.Series, eps: float = 1e-6) -> pd.Series:
    return np.log(s.astype("float64") + float(eps))


def prepare_panel_with_rival_prices(
    df: pd.DataFrame,
    products: List[str],
    qty_col: str = "qty",
    price_col: str = "avg_price",
    key_cols: Tuple[str, str] = ("date", "city"),
) -> pd.DataFrame:
    """
    Builds a wide panel at (date, city) with columns:
      - ln_qty__{product} for each selected product
      - ln_price__{product} for each selected product
    Only keeps keys where at least one selected product appears.
    """
    df = df[df["product"].isin(products)].copy()
    df["_ln_qty"] = _safe_log(df[qty_col])
    df["_ln_price"] = _safe_log(df[price_col])

    # Pivot to wide by product for ln_qty and ln_price
    base_keys = list(key_cols)
    qty_wide = df.pivot_table(index=base_keys, columns="product", values="_ln_qty", aggfunc="sum")
    price_wide = df.pivot_table(index=base_keys, columns="product", values="_ln_price", aggfunc="mean")

    # Align columns order
    qty_wide = qty_wide.reindex(columns=products)
    price_wide = price_wide.reindex(columns=products)

    # Merge back to one frame
    qty_wide.columns = [f"ln_qty__{c}" for c in qty_wide.columns]
    price_wide.columns = [f"ln_price__{c}" for c in price_wide.columns]
    panel = pd.concat([qty_wide, price_wide], axis=1).dropna(how="all", subset=qty_wide.columns)

    return panel.reset_index()


def fit_cross_price_for_focal(
    panel: pd.DataFrame,
    focal: str,
    alpha: float = 1.0,
    min_obs: int = 80,
) -> Optional[CrossPriceResult]:
    """
    Fits a ridge regression for a focal product:
        ln_qty_focal ~ ln_price_focal + ln_price_rival1 + ... + time/city FE (optional upstream)
    Returns elasticities for own and rival prices.
    """
    y_col = f"ln_qty__{focal}"
    price_cols = [c for c in panel.columns if c.startswith("ln_price__")]
    if y_col not in panel.columns:
        return None

    # Keep rows where y is observed
    work = panel.dropna(subset=[y_col]).copy()
    if len(work) < min_obs:
        return None

    # Design matrix: all price columns; drop all-NaN columns
    X = work[price_cols].copy()
    X = X.dropna(axis=1, how="all")
    # Align y to rows with at least own price observed
    y = work[y_col]

    # If very sparse, bail out
    if X.shape[1] == 0:
        return None

    # Ridge for stability; report coefs
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    mask = X.notna().all(axis=1)
    X_fit = X.loc[mask]
    y_fit = y.loc[mask]
    if len(y_fit) < min_obs:
        return None

    model.fit(X_fit.values, y_fit.values)
    coefs = pd.Series(model.coef_, index=X_fit.columns)

    # Map to tidy names
    tidy = {}
    for k, v in coefs.items():
        prod = k.replace("ln_price__", "")
        name = "own" if prod == focal else prod
        tidy[name] = float(v)

    # R^2 (on fit rows)
    y_hat = model.predict(X_fit.values)
    r2 = float(1.0 - np.var(y_fit.values - y_hat) / np.var(y_fit.values)) if y_fit.var() > 0 else np.nan

    return CrossPriceResult(
        focal=focal,
        coefs=pd.Series(tidy).sort_index(),
        r2=r2,
        nobs=int(len(y_fit)),
    )


def cross_price_matrix(
    panel: pd.DataFrame,
    products: List[str],
    alpha: float = 1.0,
    min_obs: int = 80,
) -> pd.DataFrame:
    """
    Fits focal-by-focal ridge models and returns a KxK elasticity matrix
    (rows = demand for product i; columns = price of product j).
    """
    rows = []
    for focal in products:
        res = fit_cross_price_for_focal(panel, focal=focal, alpha=alpha, min_obs=min_obs)
        if res is None:
            continue
        row = {"focal": focal}
        for k, v in res.coefs.items():
            col = focal if k == "own" else k
            row[col] = v
        row["r2"] = res.r2
        row["nobs"] = res.nobs
        rows.append(row)

    M = pd.DataFrame(rows).set_index("focal").reindex(products)
    # Reorder columns to match products (own on diagonal)
    ordered_cols = products.copy()
    M = M.reindex(columns=ordered_cols + [c for c in M.columns if c not in ordered_cols])
    return M
