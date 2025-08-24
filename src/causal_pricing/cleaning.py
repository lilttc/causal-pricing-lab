# src/causal_pricing/cleaning.py
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DATE_COL = "Order Date"
PRODUCT_COL = "Product"
QTY_COL = "Quantity Ordered"
PRICE_COL = "Price Each"
ADDR_COL = "Purchase Address"
REV_COL = "Revenue"

OUTLIER_IQR_MULT = 5.0


# -------------------------- private utilities -------------------------------

def _is_likely_header_row(value: object) -> bool:
    """
    Returns True if a cell looks like an accidental header token (e.g., 'Order Date')
    inside the date column after concatenation of monthly CSVs.
    """
    if not isinstance(value, str):
        return False
    return value.strip().lower().startswith("order date")


def _parse_city_state_zip(address: object) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses (city, state, zip) from addresses like:
      '669 Spruce St, Los Angeles, CA 90001'.
    Returns (None, None, None) if parsing is not possible.
    """
    if not isinstance(address, str) or "," not in address:
        return (None, None, None)

    parts = [p.strip() for p in address.split(",")]
    if len(parts) < 3:
        return (None, None, None)

    city = parts[-2]
    state_zip = parts[-1]

    m = re.search(r"([A-Z]{2})\s+(\d{5})(?:-\d{4})?$", state_zip)
    if m:
        return (city or None, m.group(1), m.group(2))

    m_state = re.search(r"\b([A-Z]{2})\b", state_zip)
    m_zip = re.search(r"\b(\d{5})(?:-\d{4})?\b", state_zip)
    return (city or None, m_state.group(1) if m_state else None, m_zip.group(1) if m_zip else None)


def _drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where all columns are NaN."""
    return df.dropna(how="all")


def _drop_repeated_headers(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """
    Removes rows that look like repeated headers based on a header-like token
    in the date column.
    """
    if date_col not in df.columns:
        return df
    mask = df[date_col].apply(_is_likely_header_row)
    return df.loc[~mask]


def _coerce_numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Converts listed columns to numeric, coercing errors to NaN."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_datetime(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Parses the datetime column with tolerant settings (errors→NaT)."""
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    return df


def _require_nonnull(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Keeps rows where all specified columns are non-null; empty if any column missing."""
    mask = np.ones(len(df), dtype=bool)
    for col in columns:
        if col in df.columns:
            mask &= df[col].notna()
        else:
            return df.iloc[0:0]
    return df.loc[mask]


def _filter_positive(df: pd.DataFrame, qty_col: str, price_col: str) -> pd.DataFrame:
    """Keeps rows with strictly positive quantity and price."""
    return df.loc[(df[qty_col] > 0) & (df[price_col] > 0)]


def _cap_outliers_iqr(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    iqr_mult: float = OUTLIER_IQR_MULT,
) -> pd.DataFrame:
    """
    Removes rows outside [Q1 - k*IQR, Q3 + k*IQR] for each column, where k is OUTLIER_IQR_MULT.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = max(0.0, q1 - iqr_mult * iqr)
        upper = q3 + iqr_mult * iqr
        df = df.loc[(df[col] >= lower) & (df[col] <= upper)]
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes exact duplicate rows."""
    return df.drop_duplicates()


def _sort_by_datetime(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Sorts by the datetime column if present."""
    if date_col not in df.columns:
        return df
    return df.sort_values(date_col)


# ---------------------------- public API -------------------------------------

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Kaggle electronics sales data in a deterministic sequence.

    Steps:
      1) Drop all-NaN rows.
      2) Remove repeated header rows (date column token).
      3) Coerce quantity/price to numeric.
      4) Parse order datetime.
      5) Require non-null in [Order Date, Product, Quantity Ordered, Price Each].
      6) Keep strictly positive quantity and price.
      7) Remove extreme outliers via IQR caps (OUTLIER_IQR_MULT).
      8) Drop exact duplicates.
      9) Sort by datetime.

    Returns
    -------
    pd.DataFrame
        Cleaned table with original columns preserved.
    """
    required = (DATE_COL, PRODUCT_COL, QTY_COL, PRICE_COL)

    return (
        df.pipe(_drop_all_nan_rows)
          .pipe(_drop_repeated_headers, date_col=DATE_COL)
          .pipe(_coerce_numeric, columns=(QTY_COL, PRICE_COL))
          .pipe(_parse_datetime, date_col=DATE_COL)
          .pipe(_require_nonnull, columns=required)
          .pipe(_filter_positive, qty_col=QTY_COL, price_col=PRICE_COL)
          .pipe(_cap_outliers_iqr, columns=(QTY_COL, PRICE_COL), iqr_mult=OUTLIER_IQR_MULT)
          .pipe(_drop_duplicates)
          .pipe(_sort_by_datetime, date_col=DATE_COL)
          .reset_index(drop=True)
    )


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Revenue = Quantity Ordered * Price Each
      - Date parts: year, month, day, weekday, hour
      - Address parts: city, state, zip

    Returns
    -------
    pd.DataFrame
        Input with additional engineered columns.
    """
    df = df.copy()

    if QTY_COL in df.columns and PRICE_COL in df.columns:
        df[REV_COL] = df[QTY_COL] * df[PRICE_COL]

    if DATE_COL in df.columns:
        dt = df[DATE_COL]
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["weekday"] = dt.dt.day_name()
        df["hour"] = dt.dt.hour

    parsed = df.get(ADDR_COL, pd.Series([None] * len(df)))
    parsed = parsed.map(_parse_city_state_zip)
    df["city"] = parsed.map(lambda t: t[0] if isinstance(t, tuple) else None)
    df["state"] = parsed.map(lambda t: t[1] if isinstance(t, tuple) else None)
    df["zip"] = parsed.map(lambda t: t[2] if isinstance(t, tuple) else None)

    return df


def make_daily_product_city(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    product_col: str = PRODUCT_COL,
    city_col: str = "city",
    qty_col: str = QTY_COL,
    revenue_col: str = REV_COL,
) -> pd.DataFrame:
    """
    Aggregates to daily × product × city and computes quantity-weighted average price.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-enriched transaction table.
    date_col : str
        Name of the datetime column to derive calendar date from.
    product_col : str
        Product column name.
    city_col : str
        City column name (may contain None).
    qty_col : str
        Quantity column name.
    revenue_col : str
        Revenue column name.

    Returns
    -------
    pd.DataFrame
        Columns: date, product, city, qty, revenue, avg_price
    """
    if date_col not in df.columns:
        raise ValueError(f"Expected datetime column '{date_col}' in dataframe.")
    if product_col not in df.columns:
        raise ValueError(f"Expected product column '{product_col}' in dataframe.")

    tmp = df.copy()
    tmp["date"] = tmp[date_col].dt.date

    grouped = (
        tmp.groupby(["date", product_col, city_col], dropna=False)
           .agg(qty=(qty_col, "sum"), revenue=(revenue_col, "sum"))
           .reset_index()
    )
    grouped["avg_price"] = np.where(grouped["qty"] > 0, grouped["revenue"] / grouped["qty"], np.nan)
    grouped = grouped.rename(columns={product_col: "product"})
    return grouped
