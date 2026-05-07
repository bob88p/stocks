"""load/silver.py – Silver Layer (Incremental + Feature Engineering)"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from utils.config_loader import DBConfig
from utils.logger import get_logger

logger = get_logger("load.silver")


# ============================================================
# MAIN FUNCTION
# ============================================================

def build_silver_layer(db: DBConfig) -> int:
    logger.info("=" * 60)
    logger.info("SILVER LAYER - Incremental Feature Engineering")
    logger.info("=" * 60)

    # Use fast_executemany for SQL Server performance
    engine = create_engine(db.get_sqlalchemy_url(), fast_executemany=True)

    # 1. Get last processed date
    last_date = _get_last_date(engine, "silver.stock_prices_clean")
    logger.info(f"Last processed date in Silver: {last_date}")

    # 2. Extract ONLY new data
    if last_date:
        query = text("""
            SELECT 
                trade_date,
                ticker,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM bronze.stock_prices_raw
            WHERE trade_date > :last_date
        """)
        df = pd.read_sql(query, engine, params={"last_date": last_date})
    else:
        df = pd.read_sql("""
            SELECT 
                trade_date,
                ticker,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM bronze.stock_prices_raw
        """, engine)

    if df.empty:
        logger.info("No new data to process for Silver.")
        return 0

    # 3. Feature Engineering (ALL IN PYTHON)
    logger.info("Calculating technical indicators...")
    df = _add_features(df)

    # 4. Deduplicate inside Silver (safety layer)
    df = df.drop_duplicates(subset=["trade_date", "ticker"])

    # 5. Load to Silver (append-only safe)
    logger.info(f"Loading {len(df)} rows into silver.stock_prices_clean...")

    # Default method with fast_executemany is best for SQL Server
    df.to_sql(
        "stock_prices_clean",
        engine,
        schema="silver",
        if_exists="append",
        index=False
    )

    engine.dispose()
    logger.info("Silver layer updated successfully [OK]")
    return len(df)


# ============================================================
# FEATURES ENGINEERING
# ============================================================

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ticker", "trade_date"])

    g = df.groupby("ticker")

    # Daily return
    df["prev_close"] = g["close_price"].shift(1)
    df["daily_return_pct"] = (df["close_price"] - df["prev_close"]) / df["prev_close"]

    # Price range
    df["price_range"] = df["high_price"] - df["low_price"]

    # Moving average (7)
    df["ma_7"] = g["close_price"].rolling(7).mean().reset_index(level=0, drop=True)

    # Volatility (7 days)
    df["volatility_7d"] = g["daily_return_pct"].rolling(7).std().reset_index(level=0, drop=True)

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop(columns=["prev_close"])

    return df


# ============================================================
# HELPERS
# ============================================================

def _get_last_date(engine: Engine, table: str):
    try:
        query = f"SELECT MAX(trade_date) FROM {table}"
        with engine.connect() as conn:
            return conn.execute(text(query)).scalar()
    except Exception:
        return None


# ============================================================
# GOLD LAYER (VIEW)
# ============================================================

def refresh_gold_view(db: DBConfig):
    logger.info("Refreshing GOLD view...")
    engine = create_engine(db.get_sqlalchemy_url())

    query = """
    CREATE OR ALTER VIEW gold.fact_stock_prices_daily AS
    SELECT
        trade_date,
        ticker,
        open_price,
        close_price,
        daily_return_pct,
        price_range,
        ma_7,
        volatility_7d
    FROM silver.stock_prices_clean;
    """

    with engine.begin() as conn:
        conn.execute(text(query))

    engine.dispose()
    logger.info("Gold view refreshed [OK]")