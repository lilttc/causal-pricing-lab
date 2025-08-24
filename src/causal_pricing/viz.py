from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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