import pandas as pd
import yfinance as yf
from utils.logger import get_logger

logger = get_logger("extract.reader")

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]


def read_all(
    symbols: list[str] | None = None,
    period: str = "3y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:

    tickers = symbols or DEFAULT_TICKERS

    logger.info("=" * 55)
    logger.info("EXTRACT LAYER — yfinance download")
    logger.info(f"  Tickers  : {tickers}")
    logger.info(f"  Period   : {period}  |  Interval: {interval}")
    logger.info("=" * 55)

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=auto_adjust,
        group_by="ticker"
    )

    if raw.empty:
        logger.warning("yfinance returned empty DataFrame")
        return pd.DataFrame(columns=[
            "trade_date", "ticker",
            "open_price", "high_price", "low_price",
            "close_price", "volume"
        ])

    # =========================
    # Multi-ticker case
    # =========================
    if isinstance(raw.columns, pd.MultiIndex):

        df = raw.stack(level=0).reset_index()

        # normalize ticker column
        df = df.rename(columns={
            "level_1": "ticker",
            "Ticker": "ticker",
            "level_0": "trade_date"
        })

    # =========================
    # Single ticker case
    # =========================
    else:
        df = raw.reset_index()
        df["ticker"] = tickers[0]

    # =========================
    # Normalize columns
    # =========================
    df.columns = [
        c.strip().lower() if isinstance(c, str) else c
        for c in df.columns
    ]

    # rename to clean schema
    rename_map = {
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "volume": "volume",
        "date": "trade_date",
        "datetime": "trade_date",
    }

    df = df.rename(columns=rename_map)

    # ensure ticker exists
    if "ticker" not in df.columns:
        raise ValueError(f"Missing 'ticker'. Columns: {df.columns}")

    # final schema
    keep = [
        "trade_date", "ticker",
        "open_price", "high_price", "low_price",
        "close_price", "volume"
    ]

    df = df[[c for c in keep if c in df.columns]].copy()

    # logging
    logger.info(f"Extracted {len(df)} rows")
    logger.info(f"Tickers: {df['ticker'].nunique()}")
    logger.info(f"Date range: {df['trade_date'].min()} -> {df['trade_date'].max()}")

    return df