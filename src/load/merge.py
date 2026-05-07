"""load/merge.py – MERGE Stage Layer into Bronze Layer.

Strategy (Upsert / SCD Type 1):
    - Source : stage.stage_layer     (fresh data from API)
    - Target : bronze.stock_prices_raw (historical data)

    WHEN MATCHED     → UPDATE  (row exists, refresh values)
    WHEN NOT MATCHED → INSERT  (new row, add it)

Why MERGE and not simple INSERT:
    - Airflow can re-run the pipeline → MERGE prevents duplicates
    - Bronze is the permanent history layer — must stay clean
    - Stage is the risky zone; MERGE is the safe gate into Bronze

This is the idempotency protection of the entire pipeline.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dataclasses import dataclass
from typing import Optional

from utils.config_loader import DBConfig
from utils.logger import get_logger

logger = get_logger("load.merge")

# ── Table references ──────────────────────────────────────────
STAGE_SCHEMA  = "stage"
STAGE_TABLE   = "stock_stage"

BRONZE_SCHEMA = "bronze"
BRONZE_TABLE  = "stock_prices_raw"

STAGE_FULL    = f"{STAGE_SCHEMA}.{STAGE_TABLE}"
BRONZE_FULL   = f"{BRONZE_SCHEMA}.{BRONZE_TABLE}"


# ============================================================
# Result dataclass — returned to pipeline.py
# ============================================================

@dataclass
class MergeResult:
    rows_in_stage:   int
    rows_inserted:   int
    rows_updated:    int
    rows_in_bronze:  int           # total rows in bronze after merge
    min_date_bronze: Optional[str]
    max_date_bronze: Optional[str]

    @property
    def total_affected(self) -> int:
        return self.rows_inserted + self.rows_updated


# ============================================================
# Main entry point — called by pipeline.py
# ============================================================

def merge_stage_to_bronze(db: DBConfig) -> MergeResult:
    """
    MERGE rows from stage.stage_layer into bronze.stock_prices_raw.

    Args:
        db: DBConfig with SQL Server credentials

    Returns:
        MergeResult with counts of inserted / updated rows

    Raises:
        RuntimeError : if merge fails
    """
    logger.info("=" * 55)
    logger.info("BRONZE LAYER — Merge from Stage")
    logger.info(f"  Source : {STAGE_FULL}")
    logger.info(f"  Target : {BRONZE_FULL}")
    logger.info("=" * 55)

    engine = _get_engine(db)

    # --- Pre-merge audit ----------------------------------------
    stage_count  = _count_rows(engine, STAGE_FULL)
    bronze_before = _count_rows(engine, BRONZE_FULL)
    logger.info(f"Stage rows        : {stage_count}")
    logger.info(f"Bronze rows before: {bronze_before}")

    if stage_count == 0:
        logger.warning("Stage layer is empty — skipping merge. Nothing to load.")
        engine.dispose()
        min_d, max_d = _get_date_range(engine, BRONZE_FULL)
        return MergeResult(
            rows_in_stage   = 0,
            rows_inserted   = 0,
            rows_updated    = 0,
            rows_in_bronze  = bronze_before,
            min_date_bronze = min_d,
            max_date_bronze = max_d,
        )

    # --- Execute MERGE -----------------------------------------
    inserted, updated = _run_merge(engine)

    # --- Post-merge audit --------------------------------------
    bronze_after     = _count_rows(engine, BRONZE_FULL)
    min_date, max_date = _get_date_range(engine, BRONZE_FULL)

    result = MergeResult(
        rows_in_stage   = stage_count,
        rows_inserted   = inserted,
        rows_updated    = updated,
        rows_in_bronze  = bronze_after,
        min_date_bronze = min_date,
        max_date_bronze = max_date,
    )

    logger.info("MERGE completed ✓")
    logger.info(f"  Inserted : {inserted} new rows")
    logger.info(f"  Updated  : {updated} existing rows")
    logger.info(f"  Bronze rows after: {bronze_after}")
    logger.info(f"  Date range: {min_date} → {max_date}")

    engine.dispose()
    return result


# ============================================================
# Private helpers
# ============================================================

def _get_engine(db: DBConfig) -> Engine:
    """Create and test SQLAlchemy engine."""
    try:
        engine = create_engine(db.get_sqlalchemy_url(), fast_executemany=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise RuntimeError(f"DB connection failed: {e}") from e


def _count_rows(engine: Engine, full_table: str) -> int:
    """Return row count of a table. Returns 0 if table doesn't exist."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table}"))
            return int(result.scalar())
    except Exception:
        return 0


def _get_date_range(engine: Engine, full_table: str):
    """Return (min_date, max_date) from a table."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MIN(trade_date), MAX(trade_date) FROM {full_table}")
            ).fetchone()
        return str(row[0]) if row[0] else None, str(row[1]) if row[1] else None
    except Exception:
        return None, None


def _run_merge(engine: Engine):
    """
    Execute the SQL Server MERGE statement.

    Returns:
        (rows_inserted, rows_updated) — estimated from row counts.

    SQL Server MERGE does not return separate inserted/updated counts directly,
    so we use OUTPUT clause to track them.
    """

    # ── MERGE with OUTPUT clause to count INSERTs vs UPDATEs ──
    merge_sql = f"""
    SET NOCOUNT ON;
    DECLARE @MergeOutput TABLE (
        action      NVARCHAR(10),
        trade_date  DATE,
        ticker      NVARCHAR(10)
    );

    MERGE {BRONZE_FULL} AS target
    USING {STAGE_FULL}  AS source
        ON  target.trade_date = source.trade_date
        AND target.ticker     = source.ticker

    -- Row exists in Bronze → UPDATE with latest values from Stage
    WHEN MATCHED THEN
        UPDATE SET
            target.open_price   = source.open_price,
            target.high_price   = source.high_price,
            target.low_price    = source.low_price,
            target.close_price  = source.close_price,
            target.volume       = source.volume

    -- Row is new → INSERT into Bronze
    WHEN NOT MATCHED BY TARGET THEN
        INSERT (trade_date, ticker, open_price, high_price, low_price, close_price, volume)
        VALUES (
            source.trade_date,
            source.ticker,
            source.open_price,
            source.high_price,
            source.low_price,
            source.close_price,
            source.volume
        )

    OUTPUT
        $action,
        inserted.trade_date,
        inserted.ticker
    INTO @MergeOutput;

    -- Return counts per action
    SELECT action, COUNT(*) AS cnt
    FROM   @MergeOutput
    GROUP  BY action;
    """

    inserted = 0
    updated  = 0

    try:
        with engine.begin() as conn:           # auto-commit on success
            results = conn.execute(text(merge_sql))
            for row in results:
                action = str(row[0]).upper()
                count  = int(row[1])
                if action == "INSERT":
                    inserted = count
                elif action == "UPDATE":
                    updated = count

        logger.debug(f"MERGE SQL executed: {inserted} inserts, {updated} updates")
        return inserted, updated

    except Exception as e:
        logger.error(f"MERGE failed: {e}")
        raise RuntimeError(f"MERGE stage→bronze failed: {e}") from e
