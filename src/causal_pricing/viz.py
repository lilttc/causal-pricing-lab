from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def barh_elasticities(df: pd.DataFrame, coef_col: str = "coef", label_col: str = "group", top: int = 20):
    """
    Draws a horizontal bar chart for elasticities.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing at least [label_col, coef_col].
    coef_col : str, default="coef"
        Column with elasticity values (negative = elastic).
    label_col : str, default="group"
        Column with labels (e.g., product names).
    top : int, default=20
        Number of rows to display after sorting ascending by coef_col.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object for further customization.
    """
    s = df.sort_values(coef_col).set_index(label_col)[coef_col].head(top)
    ax = s.plot(kind="barh", figsize=(6, max(3, int(top * 0.4))))
    ax.set_xlabel("Own-price elasticity (β)")
    ax.set_ylabel(label_col.title())
    ax.axvline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    return ax


def barh_elasticities_from_products(df: pd.DataFrame, product_col: str = "product", elasticity_col: str = "elasticity", top: int = 20):
    """
    Convenience wrapper for tables shaped like:
        [product, elasticity, ...]
    Useful for both baseline (OLS) and DML results.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain [product_col, elasticity_col].
    product_col : str, default="product"
        Column with product names.
    elasticity_col : str, default="elasticity"
        Column with elasticity values.
    top : int, default=20
        Number of rows to display after sorting ascending.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object.
    """
    plot_df = (
        df.rename(columns={product_col: "group", elasticity_col: "coef"})
          .loc[:, ["group", "coef"]]
    )
    return barh_elasticities(plot_df, coef_col="coef", label_col="group", top=top)


def compare_elasticities(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "product",
    left_name: str = "OLS",
    right_name: str = "DML",
    elasticity_col: str = "elasticity",
    top: int = 20,
):
    """
    Plots a side-by-side comparison of two elasticity tables (e.g., OLS vs DML).
    The function merges on a key column (default 'product'), computes the
    difference, and displays a paired horizontal plot.

    Parameters
    ----------
    left : pd.DataFrame
        First table with [on, elasticity_col].
    right : pd.DataFrame
        Second table with [on, elasticity_col].
    on : str, default="product"
        Key column to merge on.
    left_name : str, default="OLS"
        Label for the left source.
    right_name : str, default="DML"
        Label for the right source.
    elasticity_col : str, default="elasticity"
        Column name holding elasticity values in both tables.
    top : int, default=20
        Number of rows to display after sorting by the right-hand elasticity.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object.
    """
    cols = [on, elasticity_col]
    L = left.loc[:, cols].rename(columns={elasticity_col: f"{left_name}"})
    R = right.loc[:, cols].rename(columns={elasticity_col: f"{right_name}"})
    merged = L.merge(R, on=on, how="inner")
    merged = merged.sort_values(right_name).head(top)

    ax = merged.set_index(on)[[left_name, right_name]].plot(
        kind="barh", figsize=(7.5, max(3, int(top * 0.45)))
    )
    ax.set_xlabel("Own-price elasticity (β)")
    ax.set_ylabel(on.title())
    ax.axvline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    return ax

def heatmap_cross_price(
    M: pd.DataFrame,
    products: list[str] | None = None,
    title: str | None = "Cross-Price Elasticities (rows: demand_i, cols: price_j)",
    fmt: str = ".2f",
):
    """
    Draws a heatmap for a K×K cross-price elasticity matrix.

    Parameters
    ----------
    M : pd.DataFrame
        Matrix with index = focal (demand for i) and columns = price of j.
    products : list[str] | None
        Optional ordered product list to select/reorder rows/cols.
    title : str | None
        Title to display above the heatmap.
    fmt : str
        Number format for annotations.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object.
    """
    if products is not None:
        M = M.reindex(index=products, columns=products)

    k = len(M.index)
    fig_w = max(6.0, 0.9 * k)
    fig_h = max(5.0, 0.85 * k)

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(M, annot=True, fmt=fmt, cmap="coolwarm", center=0, square=True)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Price of j")
    ax.set_ylabel("Demand for i")
    plt.tight_layout()
    return ax

def bar_revenue_change(df: pd.DataFrame, value_col: str = "d_rev_abs", label_col: str = "product", top: int = 20):
    """
    Horizontal bar chart of revenue change by product for a scenario table produced by ScenarioResult.to_frame().

    Parameters
    ----------
    df : pd.DataFrame
        Must include [label_col, value_col].
    value_col : str, default="d_rev_abs"
        Absolute revenue change per product.
    label_col : str, default="product"
        Product label column.
    top : int, default=20
        Number of rows to display after sorting by |value_col|.

    Returns
    -------
    matplotlib.axes.Axes
    """
    d = df.copy()
    if label_col not in d.columns and "index" in d.columns:
        d = d.rename(columns={"index": label_col})
    d = d.assign(_abs=lambda x: x[value_col].abs()).sort_values("_abs", ascending=False).head(top)
    s = d.set_index(label_col)[value_col]
    ax = s.plot(kind="barh", figsize=(7, max(3, int(top * 0.45))))
    ax.set_xlabel("Δ Revenue (absolute)")
    ax.set_ylabel(label_col.title())
    ax.axvline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    return ax


def waterfall_revenue(total0: float, total1: float, contributions: pd.Series, title: str | None = None):
    """
    Simple waterfall plot showing per-product revenue contributions from baseline to scenario.

    Parameters
    ----------
    total0 : float
        Baseline total revenue.
    total1 : float
        Scenario total revenue.
    contributions : pd.Series
        Per-product absolute revenue change (should sum close to total1 - total0).
    title : str | None
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    parts = contributions.sort_values(ascending=False)
    cum = parts.cumsum()
    idx = ["Baseline"] + parts.index.tolist() + ["Scenario"]
    vals = [total0] + parts.tolist() + [total1]

    # Bar positions and colors
    x = np.arange(len(idx))
    colors = ["#999999"] + ["#2ca02c" if v >= 0 else "#d62728" for v in parts] + ["#1f77b4"]

    fig, ax = plt.subplots(figsize=(max(7, len(idx) * 0.5), 4))
    ax.bar(x[0], total0, color=colors[0])
    running = total0
    for i, v in enumerate(parts, start=1):
        running += v
        ax.bar(x[i], v, bottom=running - v, color=colors[i])
    ax.bar(x[-1], total1, color=colors[-1])
    ax.set_xticks(x)
    ax.set_xticklabels(idx, rotation=45, ha="right")
    ax.set_ylabel("Revenue")
    if title:
        ax.set_title(title)
    ax.axhline(total0, linestyle="--", linewidth=1)
    plt.tight_layout()
    return ax
