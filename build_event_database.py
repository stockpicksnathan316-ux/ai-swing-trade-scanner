"""
build_event_database.py
Phase 2, Task 1: Historical Data Collector
Collects release dates and sector ETF price data.
Saves to local cache files (not committed to git).
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path("./event_cache")
CACHE_DIR.mkdir(exist_ok=True)

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLP",
    "XLU", "XLI", "XLB", "XLY", "XLRE", "XLC"
]

# Years to collect (10+ years)
START_YEAR = 2015
END_YEAR = 2025

# API Keys — set via .env file
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

# FRED release IDs (verified stable identifiers).
# FOMC uses a hardcoded list below because FRED release_id 101 returns
# ~300 rows/year, not the ~8/yr decision dates we actually want.
# The value None for FOMC is a placeholder — the special-case branch in
# fetch_release_dates() ignores it.
FRED_RELEASES = {
    "CPI": 10,          # Consumer Price Index
    "NFP": 50,          # Employment Situation
    "FOMC": None,       # handled by special-case branch
}

# FOMC decision dates 2015-2025 (statement release date).
# Source: Federal Reserve historical calendars.
FOMC_DATES = [
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
]


# ── Utility: Save/Load Cache ───────────────────────────────────

def save_cache(name, df):
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} rows -> {path}")


def load_cache(name):
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── 1. Fetch Release Dates from FRED ───────────────────────────

def fetch_release_dates(event_name, release_id):
    """
    Fetch historical release dates from FRED.
    Returns DataFrame with columns: [date, event_type]
    """
    cache = load_cache(f"release_dates_{event_name}")
    if cache is not None:
        print(f"  [cache] Release dates for {event_name}: {len(cache)} rows")
        return cache

    # FOMC handled specially — hardcoded list, FRED release_id is unreliable.
    if event_name == "FOMC":
        df = pd.DataFrame({
            "date": pd.to_datetime(FOMC_DATES),
            "event_type": "FOMC",
        })
        print(f"  Using hardcoded FOMC dates: {len(df)} rows")
        save_cache(f"release_dates_{event_name}", df)
        return df

    print(f"  Fetching {event_name} release dates from FRED...")
    url = "https://api.stlouisfed.org/fred/release/dates"
    params = {
        "release_id": release_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "include_release_dates_with_no_data": "true",
        "limit": 1000,
    }

    all_dates = []
    offset = 0
    while True:
        params["offset"] = offset
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"    FRED error {resp.status_code}: {resp.text[:200]}")
            break
        data = resp.json()
        dates = data.get("release_dates", [])
        if not dates:
            break
        for d in dates:
            date_str = d.get("date")
            if date_str:
                all_dates.append({"date": date_str, "event_type": event_name})
        offset += 1000
        if len(dates) < 1000:
            break
        time.sleep(0.5)  # rate limit

    df = pd.DataFrame(all_dates)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= f"{START_YEAR}-01-01") &
                (df["date"] <= f"{END_YEAR}-12-31")]
        df = df.drop_duplicates(subset=["date", "event_type"]).sort_values("date")
    save_cache(f"release_dates_{event_name}", df)
    return df


# ── 2. Fetch Sector ETF Price Data ─────────────────────────────

def fetch_sector_prices():
    """
    Download daily OHLCV for all 11 SPDR sector ETFs.
    Returns long-form DataFrame: [date, ticker, open, high, low, close, volume]
    """
    cache = load_cache("sector_prices")
    if cache is not None:
        print(f"  [cache] Sector prices: {len(cache)} rows")
        return cache

    print(f"  Downloading sector ETF prices ({START_YEAR}-{END_YEAR})...")
    frames = []
    for ticker in SECTOR_ETFS:
        print(f"    {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=f"{START_YEAR}-01-01",
                end=f"{END_YEAR+1}-01-01",
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print(f"      {ticker}: no data")
                continue
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["ticker"] = ticker
            frames.append(df)
        except Exception as e:
            print(f"      {ticker} error: {e}")
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    save_cache("sector_prices", combined)
    return combined


# ── 3. Main Orchestrator ───────────────────────────────────────

def build_database():
    print("=" * 60)
    print("PHASE 2 - TASK 1: Historical Event Database Builder")
    print("=" * 60)

    # Step 1: Release dates
    print("\n[1/3] Fetching release dates from FRED...")
    release_frames = []
    for event_name, release_id in FRED_RELEASES.items():
        df = fetch_release_dates(event_name, release_id)
        if df is not None and not df.empty:
            release_frames.append(df)

    if release_frames:
        all_releases = pd.concat(release_frames, ignore_index=True)
        all_releases = all_releases.sort_values("date").reset_index(drop=True)
        save_cache("all_release_dates", all_releases)
        print(f"  Total release dates: {len(all_releases)}")
    else:
        print("  WARNING: No release dates collected.")

    # Step 2: Sector prices
    print("\n[2/3] Fetching sector ETF prices...")
    prices = fetch_sector_prices()
    if not prices.empty:
        print(f"  Total price rows: {len(prices)}")

    # Step 3: Summary
    print("\n[3/3] Summary")
    print(f"  Release dates:  {'all_release_dates.parquet' if release_frames else 'MISSING'}")
    print(f"  Consensus:      SKIPPED (no free source available)")
    print(f"  Prices:         {'sector_prices.parquet' if not prices.empty else 'MISSING'}")
    print(f"\n  Cache directory: {CACHE_DIR.resolve()}")
    print("\nDone. Run event_reactions.py next to build aggregate stats.")


if __name__ == "__main__":
    build_database()