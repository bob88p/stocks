import time
import sys
import os
import argparse
import pandas as pd
import io

from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
# Fix UTF-8 logging output (Windows CP1252 issue)
# ─────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ─────────────────────────────────────────────
# Project path setup
# ─────────────────────────────────────────────
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
from utils.config_loader import load_config
from utils.logger import get_logger
from extract.reader import read_all
from transformation.clean import clean, write_rejections
from transformation.check import check, check_stage_data
from load.load import load_to_stage
from load.merge import merge_stage_to_bronze
from load.silver import build_silver_layer, refresh_gold_view

logger = get_logger("pipeline")


# ============================================================
# Helpers
# ============================================================

def get_last_bronze_date(db):
    """Get last loaded date from Bronze for incremental extract."""
    try:
        engine = create_engine(db.get_sqlalchemy_url())
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT MAX(trade_date) FROM bronze.stock_prices_raw")
            ).scalar()
    except Exception as e:
        logger.warning(f"Could not fetch last bronze date: {e}")
        return None


# ============================================================
# PIPELINE
# ============================================================

def run(config_path: str) -> None:
    start = time.time()

    # ─────────────────────────────
    # 1. Load Config
    # ─────────────────────────────
    logger.info("=" * 60)
    logger.info(" ETL PIPELINE STARTED")
    logger.info("=" * 60)

    config = load_config(config_path)
    logger.info(f"Job: {config.job.name} v{config.job.version}")

    # ─────────────────────────────
    # 2. Extract (Incremental)
    # ─────────────────────────────
    logger.info(" Extracting data...")

    last_date = get_last_bronze_date(config.db)
    logger.info(f"Last Bronze Date: {last_date}")

    df = read_all(
        symbols=config.input.symbols,
        period=config.input.period,
        interval=config.input.interval,
        auto_adjust=config.input.auto_adjust
    )

    if df.empty:
        logger.info("No data extracted from source.")
        return

    new_data = True

    if last_date:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        last_dt = pd.to_datetime(last_date)

        df = df[df["trade_date"] > last_dt]

        new_data = not df.empty

        if new_data:
            logger.info(f"Incremental load -> {len(df)} new rows")
        else:
            logger.info("No new data to process")

    else:
        logger.info(f"Full load -> {len(df)} rows")

    # ─────────────────────────────
    # 3. ETL only if new data exists
    # ─────────────────────────────
    if new_data:

        # CLEAN
        logger.info(" Cleaning data...")
        df, rejected = clean(df)

        logger.info(f"Valid: {len(df)} | Rejected: {len(rejected)}")

        # QUALITY CHECK
        logger.info(" Pre-load validation...")
        report = check(df, stage_name="Pre-load Validation")

        if not report.is_valid:
            logger.error(f"Quality failed: {report.errors}")
            sys.exit(1)

        # LOAD TO STAGE
        logger.info(" Loading to Stage...")
        rows_loaded = load_to_stage(df, config.db)

        # SAVE REJECTIONS
        if not rejected.empty:
            path = write_rejections(rejected, config.rejection.rejection_path)
            logger.info(f"Rejected saved -> {path}")

        # POST CHECK
        logger.info(" Post-load validation...")
        stage_report = check_stage_data(config.db, rows_loaded)

        if not stage_report.is_valid:
            logger.error(f"Stage validation failed: {stage_report.errors}")
            sys.exit(1)

        # BRONZE MERGE
        logger.info(" Merging into Bronze...")
        merge_result = merge_stage_to_bronze(config.db)

        logger.info(
            f"Bronze -> Inserted: {merge_result.rows_inserted}, "
            f"Updated: {merge_result.rows_updated}"
        )

    else:
        logger.info(" Skipping ETL (no new data)")
        merge_result = None

    # ─────────────────────────────
    # 4. SILVER + GOLD (Always run safe incremental logic)
    # ─────────────────────────────
    logger.info(" Processing Silver layer...")

    silver_rows = build_silver_layer(config.db)

    if silver_rows > 0:
        logger.info(f"Silver updated -> {silver_rows} rows")
        refresh_gold_view(config.db)
        logger.info(" Gold view refreshed")
    else:
        logger.info("Silver already up to date")

    # ─────────────────────────────
    # 5. FINISH
    # ─────────────────────────────
    elapsed = round(time.time() - start, 2)

    logger.info("=" * 60)
    logger.info(" PIPELINE COMPLETED")
    logger.info(f"New records processed: {len(df) if new_data else 0}")
    logger.info(f"Silver processed: {silver_rows}")
    logger.info(f"Duration: {elapsed}s")
    logger.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/job_config.yaml")
    args = parser.parse_args()

    run(args.config)