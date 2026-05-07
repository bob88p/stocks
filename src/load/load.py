import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Optional, Tuple

from utils.config_loader import DBConfig
from utils.logger import get_logger

logger = get_logger("load.stage")

STAGE_SCHEMA = "stage"
STAGE_TABLE = "stock_stage"
FULL_TABLE = f"{STAGE_SCHEMA}.{STAGE_TABLE}"

REQUIRED_COLS = [
    "trade_date", "ticker",
    "open_price", "high_price", "low_price",
    "close_price", "volume"
]


# ============================================================
# MAIN FUNCTION
# ============================================================

def load_to_stage(df: pd.DataFrame, db: DBConfig) -> int:

    logger.info("=" * 55)
    logger.info("STAGE LAYER - Full Refresh Load")
    logger.info("=" * 55)

    if df is None or df.empty:
        logger.warning("Empty DataFrame -> skipping load")
        return 0

    engine = _get_engine(db)

    min_d, max_d, current_rows = _get_stage_info(engine)

    if current_rows > 0:
        logger.info(f"Stage BEFORE load -> {current_rows} rows | {min_d} -> {max_d}")
    else:
        logger.info("Stage is empty (first run)")

    _validate_dataframe(df)

    logger.info(
        f"Incoming -> {len(df)} rows | "
        f"{df['trade_date'].min()} -> {df['trade_date'].max()} | "
        f"tickers: {sorted(df['ticker'].unique())}"
    )

    df = _prepare(df)

    rows = _truncate_insert(df, engine)

    logger.info(f"Stage AFTER load -> {rows} rows loaded OK")

    engine.dispose()
    return rows


# ============================================================
# ENGINE
# ============================================================

def _get_engine(db: DBConfig) -> Engine:
    try:
        engine = create_engine(
            db.get_sqlalchemy_url(),
            fast_executemany=True
        )

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info(f"Connected to DB: {db.server}/{db.database}")
        return engine

    except Exception as e:
        raise RuntimeError(f"DB connection failed: {e}") from e


# ============================================================
# STAGE AUDIT
# ============================================================

def _get_stage_info(engine: Engine) -> Tuple[Optional[str], Optional[str], int]:

    query = f"""
        SELECT
            MIN(trade_date),
            MAX(trade_date),
            COUNT(*)
        FROM {FULL_TABLE}
    """

    try:
        with engine.connect() as conn:
            row = conn.execute(text(query)).fetchone()

        if row is None:
            return None, None, 0

        return row[0], row[1], int(row[2])

    except Exception as e:
        logger.warning(f"Stage not available yet: {e}")
        return None, None, 0


# ============================================================
# VALIDATION
# ============================================================

def _validate_dataframe(df: pd.DataFrame) -> None:

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["trade_date"].isna().any() or df["ticker"].isna().any():
        raise ValueError("Null values found in trade_date or ticker")

    if df.duplicated(subset=["trade_date", "ticker"]).any():
        raise ValueError("Duplicate (trade_date, ticker) found")


# ============================================================
# PREP
# ============================================================

def _prepare(df: pd.DataFrame) -> pd.DataFrame:

    df = df[REQUIRED_COLS].copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ticker"] = df["ticker"].str.upper().str.strip()

    for col in ["open_price", "high_price", "low_price", "close_price"]:
        df[col] = df[col].astype(float).round(4)

    df["volume"] = df["volume"].astype("int64")

    return df


# ============================================================
# LOAD (ATOMIC)
# ============================================================

def _truncate_insert(df: pd.DataFrame, engine: Engine) -> int:

    logger.info(f"Starting atomic load -> {len(df)} rows")

    try:
        with engine.begin() as conn:

            conn.execute(text(f"TRUNCATE TABLE {FULL_TABLE}"))

            df.to_sql(
                name=STAGE_TABLE,
                con=conn,
                schema=STAGE_SCHEMA,
                if_exists="append",
                index=False,
                chunksize=2000
            )

        return len(df)

    except Exception as e:
        raise RuntimeError(f"Stage load failed (rolled back): {e}") from e