""

import pandas as pd
import numpy as np
from datetime import date
from typing import Tuple
import os

from utils.logger import get_logger

logger = get_logger("transformation.clean")

# Price and volume columns
PRICE_COLS  = ["open_price", "high_price", "low_price", "close_price"]
NUM_COLS    = PRICE_COLS + ["volume"]
KEY_COLS    = ["trade_date", "ticker"]


# ============================================================
# Main entry point — called by pipeline.py
# ============================================================

def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("=" * 55)
    logger.info("Cleaning ")
    logger.info(f"Input rows : {len(df)}")
    logger.info("=" * 55)

    rejected_frames = []

    # --- Step 1: Standardize date format -----------------------
    df = _standardize_dates(df)

    # --- Step 2: Remove duplicates -----------------------------
    df, dup_rejected = _remove_duplicates(df)
    if not dup_rejected.empty:
        rejected_frames.append(dup_rejected)

    # --- Step 3: Handle missing values -------------------------
    #   Missing dates (weekends/holidays) are NOT in the DataFrame at all
    #   → we only handle NaN values inside existing rows
    df, null_rejected = _handle_missing_values(df)
    if not null_rejected.empty:
        rejected_frames.append(null_rejected)

    # --- Step 4: Validate numerical columns --------------------
    df, num_rejected = _validate_numerical(df)
    if not num_rejected.empty:
        rejected_frames.append(num_rejected)

    # --- Step 5: Cast final dtypes ----------------------------
    df = _cast_dtypes(df)

    # --- Step 6: Sort ------------------------------------------
    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    # --- Combine all rejections --------------------------------
    rejected_df = (
        pd.concat(rejected_frames, ignore_index=True)
        if rejected_frames
        else pd.DataFrame(columns=list(df.columns) + ["reject_reason"])
    )

    logger.info(f"Valid rows    : {len(df)}")
    logger.info(f"Rejected rows : {len(rejected_df)}")
    if len(rejected_df):
        logger.warning(f"Rejection reasons:\n{rejected_df['reject_reason'].value_counts().to_string()}")

    return df, rejected_df


# ============================================================
# Step 1 — Standardize date format
# ============================================================

def _standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert trade_date column to YYYY-MM-DD string format."""
    logger.info("Step 1 | Standardizing date format → YYYY-MM-DD")
    before = df["trade_date"].dtype

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Rows where date couldn't be parsed → will be caught in null check
    unparseable = df["trade_date"].isna().sum()
    if unparseable:
        logger.warning(f"  {unparseable} rows have unparseable dates (set to NaT)")

    logger.info(f"  trade_date dtype : {before} → str (YYYY-MM-DD)")
    return df


# ============================================================
# Step 2 — Remove duplicates
# ============================================================

def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate (trade_date, ticker) pairs — keep first occurrence."""
    logger.info("Step 2 | Removing duplicates on (trade_date, ticker)")

    dup_mask = df.duplicated(subset=KEY_COLS, keep="first")
    n_dups   = dup_mask.sum()

    if n_dups:
        logger.warning(f"  Found {n_dups} duplicate rows — removing")
        rejected = df[dup_mask].copy()
        rejected["reject_reason"] = "duplicate_date_ticker"
        df = df[~dup_mask].copy()
    else:
        logger.info("  No duplicates found ✓")
        rejected = pd.DataFrame()

    return df, rejected


# ============================================================
# Step 3 — Handle missing values
# ============================================================

def _handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle NaN values in the DataFrame.

    Strategy:
      - Date or Ticker is NaN → REJECT the row (cannot identify the record)
      - Price columns are NaN → REJECT the row (critical financial data)
      - Volume is NaN → REJECT the row (required for analytics)

    NOTE: Missing dates (weekends / market holidays) are NOT rows in the
          DataFrame — they simply don't exist. We do NOT fill or flag them.
          This is EXPECTED behavior from yfinance.
    """
    logger.info("Step 3 | Handling missing values")
    logger.info("  [INFO] Missing dates (weekends/holidays) are expected — not treated as errors")

    rejected_rows = []

    # --- Reject rows with missing trade_date or ticker ---------------
    identity_null = df[KEY_COLS].isna().any(axis=1)
    if identity_null.sum():
        bad = df[identity_null].copy()
        bad["reject_reason"] = "missing_date_or_ticker"
        rejected_rows.append(bad)
        df = df[~identity_null].copy()
        logger.warning(f"  Rejected {len(bad)} rows with null trade_date or ticker")

    # --- Reject rows with ALL price columns NaN ----------------
    all_prices_null = df[PRICE_COLS].isna().all(axis=1)
    if all_prices_null.sum():
        bad = df[all_prices_null].copy()
        bad["reject_reason"] = "all_prices_null"
        rejected_rows.append(bad)
        df = df[~all_prices_null].copy()
        logger.warning(f"  Rejected {len(bad)} rows with all prices NULL")

    # --- Reject rows with ANY critical price column NaN --------
    price_null = df[PRICE_COLS].isna().any(axis=1)
    if price_null.sum():
        bad = df[price_null].copy()
        bad["reject_reason"] = "price_column_null"
        rejected_rows.append(bad)
        df = df[~price_null].copy()
        logger.warning(f"  Rejected {len(bad)} rows with NULL price values")

    # --- Reject rows where volume is NaN -----------------------
    vol_null = df["volume"].isna()
    if vol_null.sum():
        bad = df[vol_null].copy()
        bad["reject_reason"] = "volume_null"
        rejected_rows.append(bad)
        df = df[~vol_null].copy()
        logger.warning(f"  Rejected {len(bad)} rows with NULL volume")

    if not rejected_rows:
        logger.info("  No missing value issues found ✓")

    rejected = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    return df, rejected


# ============================================================
# Step 4 — Validate numerical columns
# ============================================================

def _validate_numerical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate price and volume columns.

    Rules:
      - All prices (Open, High, Low, Close) must be > 0
      - Volume must be >= 0
      - High >= Low  (basic price sanity check)
      - High >= Open, High >= Close
      - Low  <= Open, Low  <= Close
    """
    logger.info("Step 4 | Validating numerical columns")
    rejected_rows = []

    # --- Prices must be > 0 ------------------------------------
    neg_price = (df[PRICE_COLS] <= 0).any(axis=1)
    if neg_price.sum():
        bad = df[neg_price].copy()
        bad["reject_reason"] = "price_zero_or_negative"
        rejected_rows.append(bad)
        df = df[~neg_price].copy()
        logger.warning(f"  Rejected {len(bad)} rows with price <= 0")

    # --- volume must be >= 0 -----------------------------------
    neg_vol = df["volume"] < 0
    if neg_vol.sum():
        bad = df[neg_vol].copy()
        bad["reject_reason"] = "volume_negative"
        rejected_rows.append(bad)
        df = df[~neg_vol].copy()
        logger.warning(f"  Rejected {len(bad)} rows with volume < 0")

    # --- high_price must be >= low_price -----------------------------------
    high_lt_low = df["high_price"] < df["low_price"]
    if high_lt_low.sum():
        bad = df[high_lt_low].copy()
        bad["reject_reason"] = "high_less_than_low"
        rejected_rows.append(bad)
        df = df[~high_lt_low].copy()
        logger.warning(f"  Rejected {len(bad)} rows where high_price < low_price")

    # --- high_price must be >= open_price and close_price ------------------------
    high_lt_open  = df["high_price"] < df["open_price"]
    high_lt_close = df["high_price"] < df["close_price"]
    bad_high = high_lt_open | high_lt_close
    if bad_high.sum():
        bad = df[bad_high].copy()
        bad["reject_reason"] = "high_less_than_open_or_close"
        rejected_rows.append(bad)
        df = df[~bad_high].copy()
        logger.warning(f"  Rejected {len(bad)} rows where high_price < open_price or close_price")

    # --- low_price must be <= open_price and close_price -------------------------
    low_gt_open  = df["low_price"] > df["open_price"]
    low_gt_close = df["low_price"] > df["close_price"]
    bad_low = low_gt_open | low_gt_close
    if bad_low.sum():
        bad = df[bad_low].copy()
        bad["reject_reason"] = "low_greater_than_open_or_close"
        rejected_rows.append(bad)
        df = df[~bad_low].copy()
        logger.warning(f"  Rejected {len(bad)} rows where low_price > open_price or close_price")

    if not rejected_rows:
        logger.info("  All numerical validations passed ✓")

    rejected = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    return df, rejected


# ============================================================
# Step 5 — Cast final dtypes
# ============================================================

def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce correct data types before loading to SQL Server."""
    df["trade_date"]   = pd.to_datetime(df["trade_date"])          # datetime64 for SQL
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["open_price"]   = df["open_price"].astype(float).round(4)
    df["high_price"]   = df["high_price"].astype(float).round(4)
    df["low_price"]    = df["low_price"].astype(float).round(4)
    df["close_price"]  = df["close_price"].astype(float).round(4)
    df["volume"] = df["volume"].astype("Int64")
    return df


def write_rejections (df_rejected, output_path):
    rej_path = os.path.join(output_path, "rejections.csv")
    df_rejected.to_csv(rej_path, index=False)
    logger.info(f"Rejected records written to: {rej_path}")
    return rej_path
