"""
src/preprocessing/cleaning.py
Data hygiene and median imputation for standardized flow tables.
"""

from typing import Dict, Optional
import polars as pl

def clean_and_impute_flows(
    df: pl.DataFrame,
    medians: Optional[Dict[str, float]] = None
) -> pl.DataFrame:
    """
    Imputes null values using pre-fitted training medians.
    Does NOT fabricate extreme artificial numbers for infinities (already null-converted).
    """
    if medians is None:
        medians = {}

    fill_exprs = []
    for col in df.columns:
        if col in medians:
            med_val = medians[col]
            fill_exprs.append(pl.col(col).fill_null(med_val))

    if fill_exprs:
        df = df.with_columns(fill_exprs)

    # Ensure timestamp is non-null
    df = df.filter(pl.col("timestamp").is_not_null())
    return df
