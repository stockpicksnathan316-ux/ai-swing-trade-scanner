"""
SEC EDGAR data fetcher for Tick Sniper.

Fetches 10+ years of annual financials directly from the SEC's free XBRL API
at data.sec.gov. No third-party packages required.

Usage:
    from sec_data import fetch_sec_financials
    data = fetch_sec_financials('AAPL')
    if data:
        revenue   = data['revenue']       # pandas Series, most recent first
        net_income = data['net_income']
        fcf       = data['fcf']           # operating cash flow - capex
        shares    = data['shares']
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# CONFIG — CHANGE THE EMAIL TO YOURS
# ============================================================
# SEC requires a User-Agent that identifies your app + a contact email.
# Requests without this are silently blocked.
SEC_USER_AGENT = "TickSniper/1.0 (contact: stockpicksnathan316@gmail.com)"

# SEC rate limit is 10 requests/second. This is a safe margin.
MIN_REQUEST_INTERVAL = 0.15

# Cache config
CACHE_DIR = Path(__file__).parent / "sec_cache"
CACHE_DIR.mkdir(exist_ok=True)
TICKERS_INDEX_FILE = CACHE_DIR / "company_tickers.json"
TICKERS_INDEX_MAX_AGE_HOURS = 24
FACTS_CACHE_MAX_AGE_DAYS = 7

# ============================================================
# XBRL TAG FALLBACKS
# ============================================================
# Companies use different XBRL tags for the same concept. We try
# each in order and use the first that returns usable data.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]

NET_INCOME_TAGS = [
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "ProfitLoss",
]

OCF_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]

CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]

SHARES_TAGS = [
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]


# ============================================================
# HTTP helper with rate limiting
# ============================================================
_last_request_time = 0.0

def _get(url):
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    r = requests.get(url, headers=headers, timeout=20)
    _last_request_time = time.time()
    r.raise_for_status()
    return r


# ============================================================
# Ticker -> CIK mapping (cached for 24h)
# ============================================================
def _load_tickers_index():
    stale = True
    if TICKERS_INDEX_FILE.exists():
        age = datetime.now() - datetime.fromtimestamp(TICKERS_INDEX_FILE.stat().st_mtime)
        stale = age > timedelta(hours=TICKERS_INDEX_MAX_AGE_HOURS)
    if stale:
        try:
            r = _get("https://www.sec.gov/files/company_tickers.json")
            TICKERS_INDEX_FILE.write_bytes(r.content)
        except Exception:
            if not TICKERS_INDEX_FILE.exists():
                raise
    with open(TICKERS_INDEX_FILE, "r") as f:
        return json.load(f)


def get_cik(ticker):
    """Return zero-padded 10-digit CIK for a US-listed ticker, or None."""
    try:
        idx = _load_tickers_index()
    except Exception:
        return None
    t = ticker.upper()
    for entry in idx.values():
        if entry.get("ticker", "").upper() == t:
            return f"{entry['cik_str']:010d}"
    return None


# ============================================================
# Company facts (cached for 7 days)
# ============================================================
def _load_company_facts(cik):
    cache_file = CACHE_DIR / f"facts_{cik}.json"
    if cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age < timedelta(days=FACTS_CACHE_MAX_AGE_DAYS):
            with open(cache_file, "r") as f:
                return json.load(f)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = _get(url)
    cache_file.write_bytes(r.content)
    return json.loads(r.content)


# ============================================================
# Extraction helpers
# ============================================================
def _extract_annual_series(facts, tags):
    """
    Try each XBRL tag in order, return the first Series that has >=3 years
    of annual (10-K, FY, ~365-day period) data.
    Indexed by fiscal year end date, most recent first.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag not in gaap:
            continue
        units = gaap[tag].get("units", {})
        for unit_key in ("USD", "shares"):
            entries = units.get(unit_key, [])
            if not entries:
                continue
            # Keep only annual entries: 10-K form, FY period, ~365-day duration
            annual = []
            for e in entries:
                if e.get("form") != "10-K":
                    continue
                if e.get("fp") != "FY":
                    continue
                if not e.get("start") or not e.get("end"):
                    continue
                try:
                    start = datetime.strptime(e["start"], "%Y-%m-%d")
                    end = datetime.strptime(e["end"], "%Y-%m-%d")
                    duration_days = (end - start).days
                except Exception:
                    continue
                # Annual periods are ~365 days (allow 300–400 for 52/53-week years)
                if not (300 <= duration_days <= 400):
                    continue
                annual.append(e)

            if not annual:
                continue

            # Deduplicate by end date, keeping the latest filed
            by_end = {}
            for e in annual:
                end = e["end"]
                filed = e.get("filed", "")
                if end not in by_end or filed > by_end[end].get("filed", ""):
                    by_end[end] = e

            items = sorted(by_end.items(), key=lambda kv: kv[0], reverse=True)
            series = pd.Series(
                {pd.Timestamp(k): float(v["val"]) for k, v in items}
            )
            if len(series) >= 3:
                return series
    return None


# ============================================================
# Main fetcher
# ============================================================
def fetch_sec_financials(ticker):
    """
    Fetch 10+ years of annual financials for a US-listed ticker.

    Returns a dict:
        {
            'revenue':     pd.Series (annual, most recent first),
            'net_income':  pd.Series,
            'ocf':         pd.Series (operating cash flow),
            'capex':       pd.Series,
            'fcf':         pd.Series (ocf - abs(capex)),
            'shares':      pd.Series,
            'cik':         str,
            'entity_name': str,
        }
    Or None if the ticker isn't US-listed / has no XBRL data.
    """
    cik = get_cik(ticker)
    if not cik:
        return None

    try:
        facts = _load_company_facts(cik)
    except Exception:
        return None

    revenue = _extract_annual_series(facts, REVENUE_TAGS)
    net_income = _extract_annual_series(facts, NET_INCOME_TAGS)
    ocf = _extract_annual_series(facts, OCF_TAGS)
    capex = _extract_annual_series(facts, CAPEX_TAGS)
    shares = _extract_annual_series(facts, SHARES_TAGS)

    # Compute FCF if we have OCF and capex
    fcf = None
    if ocf is not None and capex is not None:
        # Align on the common index
        common = ocf.index.intersection(capex.index)
        if len(common) >= 3:
            fcf = ocf.loc[common] - capex.loc[common].abs()

    entity_name = facts.get("entityName", ticker)

    return {
        "revenue": revenue,
        "net_income": net_income,
        "ocf": ocf,
        "capex": capex,
        "fcf": fcf,
        "shares": shares,
        "cik": cik,
        "entity_name": entity_name,
    }