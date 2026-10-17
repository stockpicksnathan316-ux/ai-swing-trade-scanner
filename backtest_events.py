"""
backtest_events.py
Phase 2, Task 4: Rule-based event backtest

Tests simple "buy sector ETF day before event, sell on event day" strategies
using the cached data from Task 1.

Usage:
    python backtest_events.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path("./event_cache")
SPLIT_DATE = pd.Timestamp("2021-01-01")  # in-sample < SPLIT, out-of-sample >= SPLIT


# ── Data loading ───────────────────────────────────────────────

def load_prices():
    df = pd.read_parquet(CACHE_DIR / "sector_prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_release_dates():
    df = pd.read_parquet(CACHE_DIR / "all_release_dates.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Trade simulation ───────────────────────────────────────────

def simulate_event_trades(prices_df, ticker, event_dates, entry_offset, exit_offset):
    """
    prices_df:   full long-form price DataFrame from cache
    ticker:      sector ETF symbol (e.g. "XLK")
    event_dates: list of pd.Timestamp release dates
    entry_offset: int, trading days relative to event day (0 = event day close,
                  -1 = close of prior trading day, etc.)
    exit_offset:  int, same convention as entry_offset

    Returns DataFrame of trades with entry_date, exit_date, entry_px,
    exit_px, ret.
    """
    px = prices_df[prices_df["ticker"] == ticker].set_index("date")["close"].sort_index()
    trading_days = px.index

    trades = []
    for ed in pd.to_datetime(event_dates):
        ev_pos = trading_days.searchsorted(ed)
        if ev_pos >= len(trading_days):
            continue
        entry_pos = ev_pos + entry_offset
        exit_pos = ev_pos + exit_offset
        if entry_pos < 0 or exit_pos < 0:
            continue
        if entry_pos >= len(trading_days) or exit_pos >= len(trading_days):
            continue
        if exit_pos <= entry_pos:
            continue

        entry_date = trading_days[entry_pos]
        exit_date = trading_days[exit_pos]
        entry_px = float(px.iloc[entry_pos])
        exit_px = float(px.iloc[exit_pos])

        trades.append({
            "event_date": ed,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "ret": (exit_px - entry_px) / entry_px,
        })

    return pd.DataFrame(trades)


# ── Stats ──────────────────────────────────────────────────────

def summarize(trades, label, holding_days=2):
    if trades is None or trades.empty:
        print(f"  {label}: no trades")
        return None

    n = len(trades)
    wr = float((trades["ret"] > 0).mean())
    avg = float(trades["ret"].mean())
    med = float(trades["ret"].median())
    std = float(trades["ret"].std(ddof=1)) if n > 1 else 0.0
    total = float((1 + trades["ret"]).prod() - 1)
    best = float(trades["ret"].max())
    worst = float(trades["ret"].min())
    sharpe = (avg / std) * np.sqrt(252 / holding_days) if std > 0 else 0.0

    print(f"  {label}:")
    print(f"    Trades:        {n}")
    print(f"    Win rate:      {wr:.1%}")
    print(f"    Avg return:    {avg*100:+.3f}%")
    print(f"    Median return: {med*100:+.3f}%")
    print(f"    Std dev:       {std*100:.3f}%")
    print(f"    Best / Worst:  {best*100:+.2f}% / {worst*100:+.2f}%")
    print(f"    Cumulative:    {total*100:+.1f}%")
    print(f"    Ann. Sharpe:   {sharpe:.2f}")

    return {
        "n": n, "win_rate": wr, "avg_ret": avg, "median_ret": med,
        "std": std, "cumulative": total, "sharpe": sharpe,
    }


# ── Strategy runner ────────────────────────────────────────────

def run_strategy(prices_df, release_df, event_type, ticker,
                 entry_offset, exit_offset, holding_days=2,
                 return_stats=False):
    """
    Splits events into in-sample (<SPLIT_DATE) and out-of-sample (>=SPLIT_DATE),
    simulates both, prints summaries.
    """
    print("=" * 70)
    print(f"STRATEGY: {event_type} -> LONG {ticker}")
    print(f"  Entry: close at event+({entry_offset}) trading days")
    print(f"  Exit:  close at event+({exit_offset}) trading days")
    print("=" * 70)

    event_dates = release_df[release_df["event_type"] == event_type]["date"].tolist()
    is_dates = [d for d in event_dates if d < SPLIT_DATE]
    oos_dates = [d for d in event_dates if d >= SPLIT_DATE]

    full_trades = simulate_event_trades(
        prices_df, ticker, event_dates, entry_offset, exit_offset
    )
    is_trades = simulate_event_trades(
        prices_df, ticker, is_dates, entry_offset, exit_offset
    )
    oos_trades = simulate_event_trades(
        prices_df, ticker, oos_dates, entry_offset, exit_offset
    )

    print("\n[FULL SAMPLE]")
    full_stats = summarize(full_trades, "All events", holding_days)
    print(f"\n[IN-SAMPLE: < {SPLIT_DATE.date()}]")
    is_stats = summarize(is_trades, "IS", holding_days)
    print(f"\n[OUT-OF-SAMPLE: >= {SPLIT_DATE.date()}]")
    oos_stats = summarize(oos_trades, "OOS", holding_days)

    print()
    if return_stats:
        return full_trades, is_trades, oos_trades, {
            "full": full_stats,
            "is": is_stats,
            "oos": oos_stats,
        }
    return full_trades, is_trades, oos_trades


def main():
    print("\nPHASE 2 - TASK 4: EVENT-DRIVEN BACKTEST\n")

    prices = load_prices()
    releases = load_release_dates()

    strategies = [
        ("CPI", "XLK"),
        ("NFP", "XLE"),
        ("NFP", "XLI"),
        ("FOMC", "XLF"),
    ]

    verdict_rows = []
    for event_type, ticker in strategies:
        _, _, _, stats = run_strategy(
            prices, releases,
            event_type=event_type,
            ticker=ticker,
            entry_offset=-1,
            exit_offset=0,
            holding_days=2,
            return_stats=True,
        )
        if stats and stats.get("oos"):
            oos = stats["oos"]
            full = stats["full"] or {}
            verdict_rows.append({
                "event_type": event_type,
                "sector_etf": ticker,
                "oos_n": oos.get("n", 0),
                "oos_win_rate": oos.get("win_rate", np.nan),
                "oos_avg_ret": oos.get("avg_ret", np.nan),
                "oos_sharpe": oos.get("sharpe", np.nan),
                "full_n": full.get("n", 0),
                "full_win_rate": full.get("win_rate", np.nan),
                "full_avg_ret": full.get("avg_ret", np.nan),
                "full_sharpe": full.get("sharpe", np.nan),
            })

    verdict_df = pd.DataFrame(verdict_rows)
    out_path = CACHE_DIR / "signal_verdicts.parquet"
    verdict_df.to_parquet(out_path, index=False)
    print(f"\nSaved backtest verdicts -> {out_path}  ({len(verdict_df)} strategies)")

    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  - If a strategy's OOS win rate is close to its IS win rate (within ~5pp),
    the edge is probably real.
  - If OOS win rate collapses to ~50%, the edge was curve-fit.
  - Ann. Sharpe > 1.0 on OOS is tradeable after friction.
  - Sharpe < 0.5 is probably not worth the effort after commissions.
""")


if __name__ == "__main__":
    main()