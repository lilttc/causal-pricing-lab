from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    """
    Simulation result for a pricing scenario.

    Attributes
    ----------
    prices0 : pd.Series
        Baseline prices indexed by product.
    qty0 : pd.Series
        Baseline quantities indexed by product.
    prices1 : pd.Series
        New prices after applying percentage changes.
    qty1 : pd.Series
        New quantities implied by the elasticity matrix.
    revenue0 : pd.Series
        Baseline revenue per product (prices0 * qty0).
    revenue1 : pd.Series
        New revenue per product (prices1 * qty1).
    profit0 : Optional[pd.Series]
        Baseline profit per product (if costs provided).
    profit1 : Optional[pd.Series]
        New profit per product (if costs provided).
    pct_changes : pd.Series
        Vector of price percentage changes used in the simulation.
    """
    prices0: pd.Series
    qty0: pd.Series
    prices1: pd.Series
    qty1: pd.Series
    revenue0: pd.Series
    revenue1: pd.Series
    profit0: Optional[pd.Series]
    profit1: Optional[pd.Series]
    pct_changes: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """
        Returns a tidy per-product table with before/after metrics and deltas.
        """
        out = pd.DataFrame({
            "price0": self.prices0,
            "qty0": self.qty0,
            "rev0": self.revenue0,
            "price1": self.prices1,
            "qty1": self.qty1,
            "rev1": self.revenue1,
        })
        out["d_price_%"] = self.pct_changes.reindex(out.index)
        out["d_rev_abs"] = out["rev1"] - out["rev0"]
        out["d_rev_%"] = np.where(out["rev0"] != 0, (out["rev1"] / out["rev0"] - 1.0), np.nan)
        if (self.profit0 is not None) and (self.profit1 is not None):
            out["prof0"] = self.profit0.reindex(out.index)
            out["prof1"] = self.profit1.reindex(out.index)
            out["d_prof_abs"] = out["prof1"] - out["prof0"]
            out["d_prof_%"] = np.where(out["prof0"] != 0, (out["prof1"] / out["prof0"] - 1.0), np.nan)
        return out.sort_values("d_rev_abs", ascending=False)


# ------------------------- helpers: preparation ------------------------------

def _as_series(x: pd.Series | dict | Iterable, index: pd.Index, name: str) -> pd.Series:
    """
    Coerces input into a pandas Series aligned to 'index'.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x, index=index)  # may raise if shapes mismatch → good failure mode
    s.index = index
    s.name = name
    return s.astype(float)


def align_inputs(
    prices0: pd.Series,
    qty0: pd.Series,
    elasticity: pd.DataFrame,
    costs: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """
    Aligns baseline vectors and elasticity matrix on a common product index.
    """
    products = elasticity.index.intersection(elasticity.columns)
    prices0 = prices0.reindex(products)
    qty0 = qty0.reindex(products)
    E = elasticity.reindex(index=products, columns=products)
    C = costs.reindex(products) if costs is not None else None
    return prices0, qty0, E, C


# ------------------------- demand & revenue equations ------------------------

def demand_after_price_change(
    qty0: pd.Series,
    pct_changes: pd.Series,
    elasticity: pd.DataFrame,
    mode: str = "first_order",
) -> pd.Series:
    """
    Computes new quantities given baseline qty, percentage price changes, and an elasticity matrix.

    Parameters
    ----------
    qty0 : pd.Series
        Baseline quantities.
    pct_changes : pd.Series
        Price changes in fractional terms (e.g., -0.10 for -10%).
    elasticity : pd.DataFrame
        Matrix with rows = demand for i, columns = price of j (log–log elasticities).
    mode : {"first_order","multiplicative"}
        - "first_order": q1_i ≈ q0_i * (1 + Σ_j E_ij * Δp_j)
        - "multiplicative": q1_i = q0_i * Π_j (1 + Δp_j)^{E_ij}

    Returns
    -------
    pd.Series
        New quantities q1 aligned to qty0.index.
    """
    qty0 = qty0.astype(float)
    pct_changes = pct_changes.reindex(qty0.index).fillna(0.0).astype(float)
    E = elasticity.reindex(index=qty0.index, columns=qty0.index).fillna(0.0).astype(float)

    if mode == "first_order":
        # linearized log–log around small price changes
        dlnq = E.values @ pct_changes.values
        q1 = qty0.values * (1.0 + dlnq)
        q1 = np.maximum(q1, 0.0)  # guard
        return pd.Series(q1, index=qty0.index, name="qty1")
    elif mode == "multiplicative":
        # exact under Cobb-Douglas-like log–log form
        factors = (1.0 + pct_changes.values)[np.newaxis, :] ** E.values
        mult = factors.prod(axis=1)
        q1 = qty0.values * mult
        q1 = np.maximum(q1, 0.0)
        return pd.Series(q1, index=qty0.index, name="qty1")
    else:
        raise ValueError("mode must be 'first_order' or 'multiplicative'")


def revenue_profit_after(
    prices0: pd.Series,
    qty0: pd.Series,
    pct_changes: pd.Series,
    elasticity: pd.DataFrame,
    costs: Optional[pd.Series] = None,
    mode: str = "first_order",
) -> Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    Computes new prices, quantities, and resulting revenue/profit vectors.
    """
    prices1 = prices0 * (1.0 + pct_changes.reindex(prices0.index).fillna(0.0))
    qty1 = demand_after_price_change(qty0, pct_changes, elasticity, mode=mode)

    rev0 = prices0 * qty0
    rev1 = prices1 * qty1

    prof0 = prof1 = None
    if costs is not None:
        prof0 = (prices0 - costs) * qty0
        prof1 = (prices1 - costs) * qty1

    return prices1, qty1, rev0, rev1, prof0, prof1


# ----------------------------- public API ------------------------------------

def simulate_scenario(
    prices0: pd.Series,
    qty0: pd.Series,
    elasticity: pd.DataFrame,
    pct_changes: dict | pd.Series,
    costs: Optional[pd.Series] = None,
    mode: str = "first_order",
) -> ScenarioResult:
    """
    Simulates the effect of product-specific price changes on quantities, revenue, and profit.

    Parameters
    ----------
    prices0 : pd.Series
        Baseline prices by product.
    qty0 : pd.Series
        Baseline quantities by product.
    elasticity : pd.DataFrame
        K×K matrix with rows = demand for i, cols = price of j.
    pct_changes : dict or pd.Series
        Price changes in fractional terms. Missing entries default to 0.
    costs : Optional[pd.Series]
        Unit costs per product (needed for profit analysis).
    mode : {"first_order","multiplicative"}
        Linearized vs multiplicative response.

    Returns
    -------
    ScenarioResult
    """
    prices0, qty0, E, C = align_inputs(prices0, qty0, elasticity, costs)

    pct_changes = _as_series(pct_changes, index=prices0.index, name="d_price_%").fillna(0.0)
    p1, q1, r0, r1, pr0, pr1 = revenue_profit_after(
        prices0=prices0, qty0=qty0, pct_changes=pct_changes, elasticity=E, costs=C, mode=mode
    )

    return ScenarioResult(
        prices0=prices0, qty0=qty0, prices1=p1, qty1=q1,
        revenue0=r0, revenue1=r1, profit0=pr0, profit1=pr1,
        pct_changes=pct_changes,
    )


def lerner_optimal_price(p: float, c: float, elasticity: float) -> float:
    """
    Computes the Lerner optimal price given unit cost c and own-price elasticity e (negative).
      (p - c) / p = -1 / e  →  p* = c / (1 + 1/e)

    Parameters
    ----------
    p : float
        Current price (unused in the formula but kept for symmetry / future extensions).
    c : float
        Unit cost.
    elasticity : float
        Own-price elasticity (should be < 0).

    Returns
    -------
    float
        Optimal price p*. If elasticity >= 0 or near zero, returns np.nan.
    """
    e = float(elasticity)
    if e >= -1e-6:  # avoid division by zero / non-sensical cases
        return np.nan
    return c / (1.0 + 1.0 / e)


def grid_search_single(
    prices0: pd.Series,
    qty0: pd.Series,
    elasticity: pd.DataFrame,
    target: str,
    grid: Iterable[float] = np.linspace(-0.2, 0.2, 41),  # -20% .. +20%
    objective: str = "revenue",  # or "profit"
    costs: Optional[pd.Series] = None,
    mode: str = "first_order",
) -> pd.DataFrame:
    """
    Searches over percentage changes for one target product, holding others fixed.

    Returns
    -------
    pd.DataFrame with columns:
        change, obj_new, obj_base, obj_delta, prices1, qty1, revenue1, profit1 (if applicable)
    """
    base = simulate_scenario(prices0, qty0, elasticity, pct_changes={}, costs=costs, mode=mode)
    base_tot_rev = base.revenue0.sum()
    base_tot_prof = base.profit0.sum() if base.profit0 is not None else None

    rows = []
    for chg in grid:
        pct = pd.Series(0.0, index=prices0.index); pct[target] = float(chg)
        res = simulate_scenario(prices0, qty0, elasticity, pct, costs=costs, mode=mode)
        tot_rev = res.revenue1.sum()
        tot_prof = res.profit1.sum() if res.profit1 is not None else None

        if objective == "profit" and tot_prof is not None and base_tot_prof is not None:
            obj_new, obj_base = tot_prof, base_tot_prof
        else:
            obj_new, obj_base = tot_rev, base_tot_rev

        rows.append({
            "change": chg,
            "obj_new": float(obj_new),
            "obj_base": float(obj_base),
            "obj_delta": float(obj_new - obj_base),
        })
    return pd.DataFrame(rows).sort_values("change")
