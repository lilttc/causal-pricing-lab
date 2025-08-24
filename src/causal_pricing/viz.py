from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def barh_elasticities(df: pd.DataFrame, coef_col="coef", label_col="group", top: int = 20):
    """
    Horizontal bar chart for elasticities. Expects df with columns [label_col, coef_col].
    """
    s = df.sort_values(coef_col).set_index(label_col)[coef_col].head(top)
    ax = s.plot(kind="barh", figsize=(6, max(3, top*0.4)))
    ax.set_xlabel("Own-price elasticity (β)")
    ax.set_ylabel(label_col.title())
    ax.axvline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    return ax
