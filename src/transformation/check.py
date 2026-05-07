"""
transformation/check.py – Data Quality & Validation Utility

Provides reusable validation functions for:
    1. DataFrame checks (pre-load)
    2. Stage table checks (post-load)

Used across ETL layers:
    Extract → Clean → Stage → Bronze → Silver
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from utils.logger import get_logger
from utils.config_loader import DBConfig

logger = get_logger("transformation.check")


# ============================================================
# Report Model
# ============================================================

@dataclass
class QualityReport:
    stage_name: str
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Generic DataFrame Validation
# ============================================================

def check(
    df: pd.DataFrame,
    stage_name: str,
    expected_cols: Optional[List[str]] = None,
    pk_cols: List[str] = ["trade_date", "ticker"],
    allow_nulls: bool = False,
    date_col: str = "trade_date",
) -> QualityReport:
    """
    Perform reusable quality checks on a pandas DataFrame.

    Checks:
        - Empty dataframe
        - Required columns
        - NULL values
        - Duplicate PKs
        - Date validation

    Args:
        df             : DataFrame to validate
        stage_name     : Name of ETL stage
        expected_cols  : Required columns
        pk_cols        : Primary key columns
        allow_nulls    : Allow NULL values
        date_col       : Date column name

    Returns:
        QualityReport
    """

    report = QualityReport(stage_name=stage_name)

    logger.info("=" * 55)
    logger.info(f"DATA QUALITY CHECK — {stage_name}")
    logger.info("=" * 55)

    # --------------------------------------------------------
    # 1. Empty Check
    # --------------------------------------------------------
    if df is None or df.empty:
        msg = f"[{stage_name}] DataFrame is empty or None."

        logger.error(msg)

        report.is_valid = False
        report.errors.append(msg)

        return report

    report.metrics["row_count"] = int(len(df))

    # --------------------------------------------------------
    # 2. Schema Check
    # --------------------------------------------------------
    if expected_cols:

        missing_cols = [
            col for col in expected_cols
            if col not in df.columns
        ]

        if missing_cols:

            msg = f"[{stage_name}] Missing required columns: {missing_cols}"

            logger.error(msg)

            report.is_valid = False
            report.errors.append(msg)

    # --------------------------------------------------------
    # 3. Null Check
    # --------------------------------------------------------
    if not allow_nulls:

        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        report.metrics["null_counts"] = {
            k: int(v)
            for k, v in null_counts.to_dict().items()
        }

        if not cols_with_nulls.empty:

            msg = (
                f"[{stage_name}] Found NULL values: "
                f"{cols_with_nulls.to_dict()}"
            )

            logger.error(msg)

            report.is_valid = False
            report.errors.append(msg)

    # --------------------------------------------------------
    # 4. Duplicate Check
    # --------------------------------------------------------
    if all(col in df.columns for col in pk_cols):

        duplicate_count = int(
            df.duplicated(subset=pk_cols).sum()
        )

        report.metrics["duplicate_count"] = duplicate_count

        if duplicate_count > 0:

            msg = (
                f"[{stage_name}] Found {duplicate_count} duplicate "
                f"records based on PK columns {pk_cols}."
            )

            logger.error(msg)

            report.is_valid = False
            report.errors.append(msg)

    # --------------------------------------------------------
    # 5. Date Validation
    # --------------------------------------------------------
    if date_col in df.columns:

        try:

            temp_dates = pd.to_datetime(df[date_col])

            min_date = temp_dates.min()
            max_date = temp_dates.max()

            report.metrics["min_date"] = str(min_date.date())
            report.metrics["max_date"] = str(max_date.date())

            logger.info(
                f"[{stage_name}] Date range: "
                f"{min_date.date()} → {max_date.date()}"
            )

            # Future date warning
            now = datetime.now()

            if max_date > now:

                warning_msg = (
                    f"[{stage_name}] Future dates detected: "
                    f"{max_date.date()}"
                )

                logger.warning(warning_msg)

                report.warnings.append(warning_msg)

        except Exception as e:

            msg = f"[{stage_name}] Failed to validate dates: {e}"

            logger.error(msg)

            report.is_valid = False
            report.errors.append(msg)

    # --------------------------------------------------------
    # 6. Ticker Metrics
    # --------------------------------------------------------
    if "ticker" in df.columns:

        report.metrics["ticker_count"] = int(
            df["ticker"].nunique()
        )

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------
    if report.is_valid:

        logger.info(
            f"[{stage_name}] QUALITY CHECK PASSED ✓ "
            f"| Rows: {len(df)}"
        )

    else:

        logger.error(
            f"[{stage_name}] QUALITY CHECK FAILED ✗ "
            f"| Errors: {len(report.errors)}"
        )

    return report


# ============================================================
# Stage Table Validation (Post-Load)
# ============================================================

def check_stage_data(
    db: DBConfig,
    expected_rows: int,
    schema: str = "stage",
    table: str = "stock_stage",
) -> QualityReport:
    """
    Validate data AFTER loading into SQL Server stage layer.

    Checks:
        - Row count reconciliation
        - Duplicate PK check
        - NULL PK check
        - Date range
        - Ticker count

    Args:
        db             : DBConfig
        expected_rows  : Expected row count from DataFrame
        schema         : SQL schema
        table          : SQL table name

    Returns:
        QualityReport
    """

    report = QualityReport(stage_name="Stage Layer")

    full_table = f"{schema}.{table}"

    logger.info("=" * 55)
    logger.info(f"POST-LOAD STAGE VALIDATION — {full_table}")
    logger.info("=" * 55)

    engine = create_engine(
        db.get_sqlalchemy_url(),
        fast_executemany=True,
    )

    try:

        with engine.connect() as conn:

            # ------------------------------------------------
            # 1. Row Count Reconciliation
            # ------------------------------------------------
            row_count = conn.execute(text(f"""
                SELECT COUNT(*)
                FROM {full_table}
            """)).scalar()

            row_count = int(row_count)

            report.metrics["row_count"] = row_count

            if row_count != expected_rows:

                msg = (
                    f"Row count mismatch — "
                    f"expected {expected_rows}, got {row_count}"
                )

                logger.error(msg)

                report.is_valid = False
                report.errors.append(msg)

            else:

                logger.info(
                    f"Row count reconciliation PASSED ✓ "
                    f"({row_count} rows)"
                )

            # ------------------------------------------------
            # 2. Duplicate PK Check
            # ------------------------------------------------
            duplicate_count = conn.execute(text(f"""
                SELECT COUNT(*)
                FROM (
                    SELECT trade_date, ticker, COUNT(*) c
                    FROM {full_table}
                    GROUP BY trade_date, ticker
                    HAVING COUNT(*) > 1
                ) d
            """)).scalar()

            duplicate_count = int(duplicate_count)

            report.metrics["duplicate_count"] = duplicate_count

            if duplicate_count > 0:

                msg = (
                    f"Found {duplicate_count} duplicate "
                    f"(trade_date, ticker) records."
                )

                logger.error(msg)

                report.is_valid = False
                report.errors.append(msg)

            else:

                logger.info("Duplicate check PASSED ✓")

            # ------------------------------------------------
            # 3. NULL PK Check
            # ------------------------------------------------
            null_count = conn.execute(text(f"""
                SELECT COUNT(*)
                FROM {full_table}
                WHERE trade_date IS NULL
                   OR ticker IS NULL
            """)).scalar()

            null_count = int(null_count)

            report.metrics["null_pk_count"] = null_count

            if null_count > 0:

                msg = (
                    f"Found {null_count} rows with NULL PK values."
                )

                logger.error(msg)

                report.is_valid = False
                report.errors.append(msg)

            else:

                logger.info("NULL check PASSED ✓")

            # ------------------------------------------------
            # 4. Date Range
            # ------------------------------------------------
            date_result = conn.execute(text(f"""
                SELECT
                    MIN(trade_date),
                    MAX(trade_date)
                FROM {full_table}
            """)).fetchone()

            report.metrics["min_date"] = str(date_result[0])
            report.metrics["max_date"] = str(date_result[1])

            logger.info(
                f"Date range: "
                f"{date_result[0]} → {date_result[1]}"
            )

            # ------------------------------------------------
            # 5. Distinct Tickers
            # ------------------------------------------------
            ticker_count = conn.execute(text(f"""
                SELECT COUNT(DISTINCT ticker)
                FROM {full_table}
            """)).scalar()

            ticker_count = int(ticker_count)

            report.metrics["ticker_count"] = ticker_count

            logger.info(
                f"Distinct tickers: {ticker_count}"
            )

    except Exception as e:

        msg = f"Stage validation failed: {e}"

        logger.error(msg)

        report.is_valid = False
        report.errors.append(msg)

    finally:

        engine.dispose()

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------
    if report.is_valid:

        logger.info(
            "STAGE VALIDATION PASSED ✓"
        )

    else:

        logger.error(
            f"STAGE VALIDATION FAILED ✗ "
            f"| Errors: {len(report.errors)}"
        )

    return report
