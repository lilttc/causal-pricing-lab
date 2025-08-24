from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class OLSFit:
    """
    Container for an OLS fit with robust SEs.
    """
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    r2: float
    nobs: int


def _safe_log(x: pd.Series, eps: float) -> pd.Series:
    """
    Applies log to a nonnegative series with a small positive offset.
    """
    return np.log(x.astype("float64") + float(eps))


def prepare_loglog_data(
    df: pd.DataFrame,
    qty_col: str = "qty",
    price_col: str = "avg_price",
    eps_qty: float = 1e-6,
    eps_price: float = 1e-6,
) -> pd.DataFrame:
    """
    Prepares a log-log dataset with ln_quantity and ln_price columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain qty_col and price_col.
    qty_col : str
        Column holding nonnegative quantities.
    price_col : str
        Column holding nonnegative prices (averages per grain).
    eps_qty : float
        Small positive offset for log transform stability.
    eps_price : float
        Small positive offset for log transform stability.

    Returns
    -------
    pd.DataFrame
        Input with 'ln_qty' and 'ln_price' added.
    """
    if qty_col not in df.columns or price_col not in df.columns:
        raise ValueError("Expected quantity and price columns in dataframe.")

    out = df.copy()
    out["ln_qty"] = _safe_log(out[qty_col], eps=eps_qty)
    out["ln_price"] = _safe_log(out[price_col], eps=eps_price)
    return out


def _fit_ols(
    y: pd.Series,
    X: pd.DataFrame,
    robust: str = "HC3",
    cluster: Optional[pd.Series] = None,
) -> OLSFit:
    """
    Fits OLS with optional heteroskedastic or clustered covariance.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Design matrix (should include constant if desired).
    robust : str
        One of {'none','HC0','HC1','HC2','HC3','cluster'}.
    cluster : Optional[pd.Series]
        Cluster labels when robust == 'cluster'.

    Returns
    -------
    OLSFit
        Coefs, SEs, p-values, conf-intervals, R2, and nobs.
    """
    model = sm.OLS(y.values, X.values, missing="drop")
    res = model.fit()

    if robust.lower() == "none":
        cov_res = res
    elif robust.lower() in {"hc0", "hc1", "hc2", "hc3"}:
        cov_res = res.get_robustcov_results(cov_type=robust.upper())
    elif robust.lower() == "cluster":
        if cluster is None:
            raise ValueError("cluster labels required when robust='cluster'")
        cov_res = res.get_robustcov_results(cov_type="cluster", groups=cluster.values)
    else:
        raise ValueError("robust must be one of {'none','HC0','HC1','HC2','HC3','cluster'}")

    params = pd.Series(cov_res.params, index=X.columns, name="coef")
    bse = pd.Series(cov_res.bse, index=X.columns, name="se")
    pvals = pd.Series(cov_res.pvalues, index=X.columns, name="pval")
    ci = pd.DataFrame(cov_res.conf_int(), index=X.columns, columns=["ci_low", "ci_high"])

    return OLSFit(
        params=params,
        bse=bse,
        pvalues=pvals,
        conf_int=ci,
        r2=float(cov_res.rsquared),
        nobs=int(cov_res.nobs),
    )


def estimate_own_price_elasticity(
    df: pd.DataFrame,
    qty_col: str = "qty",
    price_col: str = "avg_price",
    controls: Optional[pd.DataFrame] = None,
    add_constant: bool = True,
    robust: str = "HC3",
    cluster_col: Optional[str] = None,
) -> Tuple[OLSFit, pd.DataFrame]:
    """
    Estimates a single log-log demand model:
        ln(qty) = alpha + beta * ln(price) + gamma' * controls + e

    The coefficient 'beta' is the own-price elasticity at the modeled grain.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain qty_col and price_col and the precomputed 'ln_' columns
        if you already called prepare_loglog_data(); otherwise they are added.
    qty_col : str
        Quantity column name.
    price_col : str
        Price column name.
    controls : Optional[pd.DataFrame]
        Controls matrix aligned to df.index (e.g., time and city dummies).
    add_constant : bool
        Whether to include an intercept in X.
    robust : str
        Covariance type ('HC3' by default).
    cluster_col : Optional[str]
        Column name to use for clustering when robust='cluster'.

    Returns
    -------
    (OLSFit, pd.DataFrame)
        The fitted model summary and the model frame used for estimation.
    """
    work = prepare_loglog_data(df, qty_col=qty_col, price_col=price_col)

    X_parts = [work[["ln_price"]]]
    if controls is not None and len(controls) > 0:
        aligned = controls.reindex(work.index)
        X_parts.append(aligned)

    X = pd.concat(X_parts, axis=1)
    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    y = work["ln_qty"]

    cluster = None
    if robust.lower() == "cluster":
        if cluster_col is None or cluster_col not in work.columns:
            raise ValueError("cluster_col must be present in df when robust='cluster'")
        cluster = work[cluster_col]

    fit = _fit_ols(y=y, X=X, robust=robust, cluster=cluster)
    return fit, pd.concat([y.rename("ln_qty"), X], axis=1)


def estimate_elasticity_by_group(
    df: pd.DataFrame,
    group_col: str = "product",
    qty_col: str = "qty",
    price_col: str = "avg_price",
    controls: Optional[pd.DataFrame] = None,
    min_obs: int = 30,
    add_constant: bool = True,
    robust: str = "HC3",
) -> pd.DataFrame:
    """
    Estimates own-price elasticity separately for each group (e.g., per product).

    Parameters
    ----------
    df : pd.DataFrame
        Data with at least group_col, qty_col, price_col.
    group_col : str
        Column to group by (e.g., 'product').
    qty_col : str
        Quantity column.
    price_col : str
        Price column.
    controls : Optional[pd.DataFrame]
        Controls matrix aligned to df.index; the same controls are used for
        every group, subset on group indices.
    min_obs : int
        Minimum observations required per group to fit a model.
    add_constant : bool
        Whether to include an intercept.
    robust : str
        Covariance type for SEs.

    Returns
    -------
    pd.DataFrame
        One row per group with columns: coef, se, pval, ci_low, ci_high, r2, nobs.
        The 'coef' column is the estimated elasticity.
    """
    rows = []
    for key, idx in df.groupby(group_col).groups.items():
        if len(idx) < int(min_obs):
            continue

        sub_df = df.loc[idx]
        sub_controls = None
        if controls is not None:
            sub_controls = controls.loc[idx]  # aligned slice

        fit, _ = estimate_own_price_elasticity(
            sub_df,
            qty_col=qty_col,
            price_col=price_col,
            controls=sub_controls,
            add_constant=add_constant,
            robust=robust,
        )

        beta = fit.params.get("ln_price", np.nan)
        se = fit.bse.get("ln_price", np.nan)
        p = fit.pvalues.get("ln_price", np.nan)
        ci = fit.conf_int.loc["ln_price"] if "ln_price" in fit.conf_int.index else pd.Series([np.nan, np.nan], index=["ci_low", "ci_high"])

        rows.append(
            dict(
                group=key,
                coef=float(beta),
                se=float(se),
                pval=float(p),
                ci_low=float(ci["ci_low"]),
                ci_high=float(ci["ci_high"]),
                r2=float(fit.r2),
                nobs=int(fit.nobs),
            )
        )

    out = pd.DataFrame(rows).sort_values("coef")
    return out.reset_index(drop=True)
