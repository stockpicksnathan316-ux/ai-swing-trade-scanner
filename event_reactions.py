"""
event_reactions.py
Phase 2, Task 2: Build Event Reaction Database

Reads raw cached data from build_event_database.py (Task 1) and computes
aggregate sector reactions per event type.

Output (in ./event_cache/):
  - reaction_stats.parquet    — unconditional stats per event × sector

Usage:
    python event_reactions.py
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path("./event_cache")

SECTOR_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLP": "Consumer Defensive",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLB": "Basic Materials",
    "XLY": "Consumer Cyclical",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

EVENT_TYPE_LABELS = {
    "CPI": "CPI (YoY)",
    "NFP": "Non-Farm Payrolls",
    "FOMC": "FOMC Rate Decision",
}


def _load_prices():
    path = CACHE_DIR / "sector_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build_event_database.py first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _load_release_dates():
    path = CACHE_DIR / "all_release_dates.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run build_event_database.py first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _event_returns_for_ticker(ticker_prices, event_dates):
    """
    ticker_prices: DataFrame with DatetimeIndex sorted ascending and 'close' col.
    event_dates:   list of pd.Timestamp release dates.
    Returns DataFrame with event-day, day-before, and day-after returns.
    """
    prices = ticker_prices.copy()
    prices["ret_1d"] = prices["close"].pct_change()
    trading_days = prices.index

    rows = []
    for ed in event_dates:
        ed = pd.Timestamp(ed)  # coerce to Timestamp (handles int/datetime/array edge cases)
        # Find first trading day on or after the release date
        idx = trading_days.searchsorted(ed)
        if idx >= len(trading_days) or idx < 1:
            continue
        trade_day = trading_days[idx]

        event_day_ret = prices["ret_1d"].iloc[idx]
        day_before_ret = prices["ret_1d"].iloc[idx - 1] if idx >= 2 else np.nan
        day_after_ret = (
            prices["ret_1d"].iloc[idx + 1] if idx + 1 < len(prices) else np.nan
        )

        rows.append({
            "event_date": ed,
            "trade_date": trade_day,
            "event_day_return": event_day_ret,
            "day_before_return": day_before_ret,
            "day_after_return": day_after_ret,
        })

    return pd.DataFrame(rows)


def build_reaction_stats():
    prices = _load_prices()
    releases = _load_release_dates()

    all_stats = []
    tickers = sorted(prices["ticker"].unique())

    for event_type in sorted(releases["event_type"].unique()):
        event_dates = (
            releases[releases["event_type"] == event_type]["date"]
            .unique()
        )
        event_dates = pd.Series(pd.to_datetime(event_dates)).sort_values().tolist()
        for ticker in tickers:
            tp = prices[prices["ticker"] == ticker].set_index("date").sort_index()
            tp["ret_1d"] = tp["close"].pct_change()
            er = _event_returns_for_ticker(tp, event_dates)
            er = er.dropna(subset=["event_day_return"])
            if er.empty:
                continue

            n = len(er)

            # Baseline: all non-event trading days for this ticker
            all_rets = tp["ret_1d"].dropna()
            non_event_returns = all_rets.drop(
                index=[d for d in er["trade_date"] if d in all_rets.index],
                errors="ignore",
            )
            baseline_avg = float(non_event_returns.mean())
            baseline_win = float((non_event_returns > 0).mean())

            all_stats.append({
                "event_type": event_type,
                "event_label": EVENT_TYPE_LABELS.get(event_type, event_type),
                "sector_etf": ticker,
                "sector_name": SECTOR_NAMES.get(ticker, ticker),
                "win_rate": float((er["event_day_return"] > 0).mean()),
                "avg_move_pct": float(er["event_day_return"].mean()),
                "median_move_pct": float(er["event_day_return"].median()),
                "avg_move_day_before_pct": float(er["day_before_return"].mean(skipna=True)),
                "avg_move_day_after_pct": float(er["day_after_return"].mean(skipna=True)),
                "baseline_avg_move_pct": baseline_avg,
                "baseline_win_rate": baseline_win,
                "lift_pct": float(er["event_day_return"].mean()) - baseline_avg,
                "sample_size": n,
            })

    df = pd.DataFrame(all_stats)
    out_path = CACHE_DIR / "reaction_stats.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Wrote {len(df)} rows -> {out_path}")
    return df


def build_all():
    print("=" * 60)
    print("PHASE 2 - TASK 2: Event Reaction Database Builder")
    print("=" * 60)

    print("\n[1/1] Unconditional reaction stats...")
    reaction_df = build_reaction_stats()

    print("\nSummary:")
    if reaction_df is not None and not reaction_df.empty:
        print(f"  reaction_stats: {len(reaction_df)} rows")
        print(f"    Event types: {sorted(reaction_df['event_type'].unique())}")
        print(f"    Sectors:     {len(reaction_df['sector_etf'].unique())} tickers")
    else:
        print("  WARNING: no reaction stats produced.")

    print("\nDone. Task 3 can now render the reaction panel.")


if __name__ == "__main__":
    build_all()