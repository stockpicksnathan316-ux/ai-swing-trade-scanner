"""
Economic calendar for Tick Sniper — self-contained, no API dependency.

Contains hardcoded 2026 and early-2027 release dates for the major US
macro events that move markets: CPI, PPI, NFP, FOMC, Retail Sales,
PCE, and GDP Advance.

DATA SOURCES (verify quarterly):
  - CPI/PPI/NFP/PCE:  https://www.bls.gov/schedule/news_release/2026_sched.htm
  - FOMC:             https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  - Retail Sales/GDP: https://www.census.gov/economic-indicators/

MAINTENANCE: Update quarterly. All times are Eastern Time (ET).

Usage:
    from econ_calendar import get_upcoming_events
    df = get_upcoming_events(days_ahead=14, impact_filter="High")
"""
import pandas as pd
from datetime import datetime, date, timedelta


# ============================================================
# EVENT DATA — 2026 + early 2027
# ============================================================
# Format: (date_str, time_str, event_name, impact, category, description)
EVENTS = [
    # ---------- OCTOBER 2026 ----------
    ("2026-10-15", "8:30 AM ET", "CPI (YoY)", "High", "Inflation",
     "Consumer Price Index — headline inflation measure"),
    ("2026-10-16", "8:30 AM ET", "PPI (MoM)", "Medium", "Inflation",
     "Producer Price Index — wholesale inflation"),
    ("2026-10-17", "8:30 AM ET", "Retail Sales (MoM)", "High", "Consumer",
     "Monthly retail spending — ~70% of US GDP"),
    ("2026-10-29", "8:30 AM ET", "GDP (Advance, Q3)", "High", "Growth",
     "First estimate of quarterly economic growth"),
    ("2026-10-30", "8:30 AM ET", "PCE (YoY)", "High", "Inflation",
     "Fed's preferred inflation gauge"),

    # ---------- NOVEMBER 2026 ----------
    ("2026-11-04", "2:00 PM ET", "FOMC Rate Decision", "High", "Fed",
     "Federal Reserve interest rate decision + press conference"),
    ("2026-11-06", "8:30 AM ET", "Non-Farm Payrolls", "High", "Employment",
     "Monthly jobs report — most market-moving data release"),
    ("2026-11-12", "8:30 AM ET", "CPI (YoY)", "High", "Inflation",
     "Consumer Price Index — headline inflation measure"),
    ("2026-11-13", "8:30 AM ET", "PPI (MoM)", "Medium", "Inflation",
     "Producer Price Index — wholesale inflation"),
    ("2026-11-18", "8:30 AM ET", "Retail Sales (MoM)", "High", "Consumer",
     "Monthly retail spending — ~70% of US GDP"),
    ("2026-11-25", "8:30 AM ET", "PCE (YoY)", "High", "Inflation",
     "Fed's preferred inflation gauge"),

    # ---------- DECEMBER 2026 ----------
    ("2026-12-04", "8:30 AM ET", "Non-Farm Payrolls", "High", "Employment",
     "Monthly jobs report — most market-moving data release"),
    ("2026-12-10", "8:30 AM ET", "CPI (YoY)", "High", "Inflation",
     "Consumer Price Index — headline inflation measure"),
    ("2026-12-11", "8:30 AM ET", "PPI (MoM)", "Medium", "Inflation",
     "Producer Price Index — wholesale inflation"),
    ("2026-12-16", "2:00 PM ET", "FOMC Rate Decision", "High", "Fed",
     "Federal Reserve interest rate decision + press conference"),
    ("2026-12-17", "8:30 AM ET", "Retail Sales (MoM)", "High", "Consumer",
     "Monthly retail spending — ~70% of US GDP"),
    ("2026-12-23", "8:30 AM ET", "PCE (YoY)", "High", "Inflation",
     "Fed's preferred inflation gauge"),

    # ---------- JANUARY 2027 ----------
    ("2027-01-08", "8:30 AM ET", "Non-Farm Payrolls", "High", "Employment",
     "Monthly jobs report — most market-moving data release"),
    ("2027-01-13", "8:30 AM ET", "CPI (YoY)", "High", "Inflation",
     "Consumer Price Index — headline inflation measure"),
    ("2027-01-14", "8:30 AM ET", "PPI (MoM)", "Medium", "Inflation",
     "Producer Price Index — wholesale inflation"),
    ("2027-01-15", "8:30 AM ET", "Retail Sales (MoM)", "High", "Consumer",
     "Monthly retail spending — ~70% of US GDP"),
    ("2027-01-27", "2:00 PM ET", "FOMC Rate Decision", "High", "Fed",
     "Federal Reserve interest rate decision + press conference"),
    ("2027-01-28", "8:30 AM ET", "GDP (Advance, Q4)", "High", "Growth",
     "First estimate of quarterly economic growth"),
    ("2027-01-29", "8:30 AM ET", "PCE (YoY)", "High", "Inflation",
     "Fed's preferred inflation gauge"),

    # ---------- FEBRUARY 2027 ----------
    ("2027-02-05", "8:30 AM ET", "Non-Farm Payrolls", "High", "Employment",
     "Monthly jobs report — most market-moving data release"),
    ("2027-02-10", "8:30 AM ET", "CPI (YoY)", "High", "Inflation",
     "Consumer Price Index — headline inflation measure"),
    ("2027-02-11", "8:30 AM ET", "PPI (MoM)", "Medium", "Inflation",
     "Producer Price Index — wholesale inflation"),
    ("2027-02-17", "8:30 AM ET", "Retail Sales (MoM)", "High", "Consumer",
     "Monthly retail spending — ~70% of US GDP"),
    ("2027-02-26", "8:30 AM ET", "PCE (YoY)", "High", "Inflation",
     "Fed's preferred inflation gauge"),

    # ---------- MARCH 2027 ----------
    ("2027-03-05", "8:30 AM ET", "Non-Farm Payrolls", "High", "Employment",
     "Monthly jobs report — most market-moving data release"),
    ("2027-03-10", "8:30 AM ET", "CPI (YoY)", "High", "Inflation",
     "Consumer Price Index — headline inflation measure"),
    ("2027-03-11", "8:30 AM ET", "PPI (MoM)", "Medium", "Inflation",
     "Producer Price Index — wholesale inflation"),
    ("2027-03-17", "2:00 PM ET", "FOMC Rate Decision", "High", "Fed",
     "Federal Reserve interest rate decision + press conference"),
    ("2027-03-17", "8:30 AM ET", "Retail Sales (MoM)", "High", "Consumer",
     "Monthly retail spending — ~70% of US GDP"),
    ("2027-03-26", "8:30 AM ET", "PCE (YoY)", "High", "Inflation",
     "Fed's preferred inflation gauge"),
]


# ============================================================
# Public API
# ============================================================
def get_upcoming_events(days_ahead=14, impact_filter=None, category_filter=None):
    """
    Return a DataFrame of upcoming economic events within `days_ahead` days
    from today.

    Args:
        days_ahead: integer, how many days into the future to look
        impact_filter: None, "High", "Medium", or "Low"
        category_filter: None, "Inflation", "Employment", "Fed",
                         "Consumer", or "Growth"

    Returns:
        pandas DataFrame with columns:
        Date, Time, Event, Impact, Category, Description, Days Until
        Sorted by date ascending. Empty DataFrame if no events match.
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    rows = []
    for entry in EVENTS:
        date_str, time_str, event, impact, category, desc = entry
        try:
            event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Skip past events
        if event_date < today:
            continue
        # Skip beyond our lookahead window
        if event_date > cutoff:
            continue
        # Apply filters
        if impact_filter and impact_filter != "All":
            if impact != impact_filter:
                continue
        if category_filter and category_filter != "All":
            if category != category_filter:
                continue

        days_until = (event_date - today).days
        if days_until == 0:
            days_label = "Today"
        elif days_until == 1:
            days_label = "Tomorrow"
        else:
            days_label = f"in {days_until}d"

        rows.append({
            "Date": date_str,
            "Time": time_str,
            "Event": event,
            "Impact": impact,
            "Category": category,
            "Description": desc,
            "Days Until": days_label,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Date", "Time", "Event", "Impact",
            "Category", "Description", "Days Until"
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Date", "Time"], ascending=[True, True]).reset_index(drop=True)
    return df


def get_next_event(event_type_keyword):
    """
    Return the single next occurrence of a given event (by keyword match).
    Useful for banner messages like 'Next CPI: Oct 15'.

    Args:
        event_type_keyword: e.g., "CPI", "FOMC", "Non-Farm"

    Returns:
        dict with Date, Time, Event, Days Until, or None if not found.
    """
    today = date.today()
    for entry in EVENTS:
        date_str, time_str, event, impact, category, desc = entry
        try:
            event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if event_date < today:
            continue
        if event_type_keyword.lower() in event.lower():
            days_until = (event_date - today).days
            return {
                "Date": date_str,
                "Time": time_str,
                "Event": event,
                "Days Until": days_until,
            }
    return None


def get_event_count():
    """Return total number of events in the database (for sanity checks)."""
    return len(EVENTS)