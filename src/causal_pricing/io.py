from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Thin wrapper around pandas.read_csv with sensible defaults.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    defaults = dict(encoding="utf-8", low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save DataFrame to Parquet. Creates parent dirs if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)