# src/causal_pricing/features.py
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _as_category(df: pd.DataFrame, col: str, out_col: Optional[str] = None) -> pd.DataFrame:
    """
    Casts a column to pandas 'category' dtype, optionally writing to a new column.
    No-op if the column is missing.
    """
    if col not in df.columns:
        return df
    out = out_col or col
    df = df.copy()
    df[out] = df[col].astype("category")
    return df


def _one_hot_with_cutoff(
    df: pd.DataFrame,
    col: str,
    min_count: int,
    prefix: Optional[str] = None,
    other_label: str = "Other",
    dtype: str = "int8",
) -> pd.DataFrame:
    """
    One-hot encodes a categorical column with rare levels grouped into 'Other'.
    Levels with frequency < min_count are mapped to 'Other' before encoding.
    """
    if col not in df.columns:
        return df

    df = df.copy()
    counts = df[col].value_counts(dropna=True)
    keep = set(counts[counts >= max(1, int(min_count))].index.tolist())

    def _norm(x):
        if pd.isna(x):
            return other_label
        return x if x in keep else other_label

    norm_col = f"{col}_norm"
    df[norm_col] = df[col].map(_norm)
    dummies = pd.get_dummies(df[norm_col], prefix=prefix or col, dtype=dtype)
    return pd.concat([df, dummies], axis=1)


# ------------------------------ public API -----------------------------------

def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds categorical time features if their source columns exist:
      - 'month_cat' from 'month'
      - 'weekday_cat' from 'weekday'
      - 'hour_cat' from 'hour'
    """
    df = df.copy()
    df = _as_category(df, "month", out_col="month_cat")
    df = _as_category(df, "weekday", out_col="weekday_cat")
    df = _as_category(df, "hour", out_col="hour_cat")
    return df


def encode_city_dummies(
    df: pd.DataFrame,
    min_count: int = 200,
    city_col: str = "city",
    prefix: str = "city",
    other_label: str = "Other",
) -> pd.DataFrame:
    """
    Adds one-hot dummies for city with a frequency cutoff. Rare cities are grouped
    into 'Other'. Produces an intermediate '<city_col>_norm' label column.
    """
    return _one_hot_with_cutoff(
        df=df,
        col=city_col,
        min_count=min_count,
        prefix=prefix,
        other_label=other_label,
        dtype="int8",
    )


def build_controls(
    df: pd.DataFrame,
    include: Iterable[str] = ("month", "weekday", "city"),
    drop_first: bool = True,
    city_min_count: int = 200,
) -> pd.DataFrame:
    """
    Builds a controls design matrix for regression models using common fixed effects.

    Parameters
    ----------
    df : pd.DataFrame
        Input table that already includes engineered time/address parts.
    include : Iterable[str]
        Which groups of controls to include from {'month','weekday','hour','city'}.
    drop_first : bool
        Whether to drop the first level per categorical set to avoid collinearity.
    city_min_count : int
        Frequency cutoff for city dummies.

    Returns
    -------
    pd.DataFrame
        Controls matrix aligned to df.index with one-hot encoded columns.
    """
    frames = []

    if "month" in include and "month" in df.columns:
        m = pd.get_dummies(df["month"].astype("category"), prefix="m", drop_first=drop_first, dtype="int8")
        frames.append(m)

    if "weekday" in include and "weekday" in df.columns:
        w = pd.get_dummies(df["weekday"].astype("category"), prefix="wd", drop_first=drop_first, dtype="int8")
        frames.append(w)

    if "hour" in include and "hour" in df.columns:
        h = pd.get_dummies(df["hour"].astype("category"), prefix="h", drop_first=drop_first, dtype="int8")
        frames.append(h)

    if "city" in include:
        tmp = encode_city_dummies(df, min_count=city_min_count)
        city_cols = [c for c in tmp.columns if c.startswith("city_")]
        if city_cols:
            C = tmp[city_cols]
            if drop_first and C.shape[1] > 0:
                C = C.iloc[:, 1:]
            frames.append(C)

    if not frames:
        return pd.DataFrame(index=df.index)

    X = pd.concat(frames, axis=1).astype("float32")
    X.index = df.index
    return X
