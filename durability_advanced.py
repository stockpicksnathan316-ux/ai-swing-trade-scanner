"""
durability_advanced.py - Full Stock Health Assessment (Durable Competitive Advantage)
Now with aggressive field name fallbacks, computed gross profit, quarterly fallback,
improved shares fallback, guarded dep_high, quarterly EPS extraction,
and earnings surprise (beat/miss) data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ------------------------------------------------------------
# Helper: assess EPS trend (moved above analyze_company)
# ------------------------------------------------------------
def assess_eps_trend_annual(net_income_series, shares_series):
    """Return 'strong', 'moderate', 'weak', or 'insufficient' based on EPS trend."""
    if net_income_series.empty or shares_series.empty:
        return 'insufficient'
    min_len = min(len(net_income_series), len(shares_series))
    if min_len < 5:
        return 'insufficient'
    eps = net_income_series.iloc[-min_len:] / shares_series.iloc[-min_len:]
    y = eps.values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    cv = y.std() / y.mean() if y.mean() != 0 else 999
    if slope > 0.05 * y.mean() and cv < 0.5:
        return 'strong'
    elif slope < -0.05 * y.mean() or cv > 1.0:
        return 'weak'
    else:
        return 'moderate'

# ------------------------------------------------------------
# Fetch annual + quarterly data
# ------------------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_all_data(ticker: str):
    """Return dict with annual and quarterly DataFrames (oldest first)."""
    stock = yf.Ticker(ticker)
    try:
        # Annual statements
        income_a = stock.income_stmt
        balance_a = stock.balance_sheet
        cashflow_a = stock.cashflow
        
        # Quarterly statements
        income_q = stock.quarterly_income_stmt
        balance_q = stock.quarterly_balance_sheet
        cashflow_q = stock.quarterly_cashflow
        
        # Reverse to oldest first
        income_a = income_a.iloc[::-1] if not income_a.empty else pd.DataFrame()
        balance_a = balance_a.iloc[::-1] if not balance_a.empty else pd.DataFrame()
        cashflow_a = cashflow_a.iloc[::-1] if not cashflow_a.empty else pd.DataFrame()
        
        income_q = income_q.iloc[::-1] if not income_q.empty else pd.DataFrame()
        balance_q = balance_q.iloc[::-1] if not balance_q.empty else pd.DataFrame()
        cashflow_q = cashflow_q.iloc[::-1] if not cashflow_q.empty else pd.DataFrame()
        
        return {
            'annual': {'income': income_a, 'balance': balance_a, 'cashflow': cashflow_a},
            'quarterly': {'income': income_q, 'balance': balance_q, 'cashflow': cashflow_q}
        }
    except Exception as e:
        st.warning(f"Data fetch error for {ticker}: {e}")
        return None

# ------------------------------------------------------------
# Main analysis – uses annual first, then quarterly fallback
# ------------------------------------------------------------
def analyze_company(ticker: str):
    data = fetch_all_data(ticker)
    if data is None:
        return None
    
    annual = data['annual']
    quarterly = data['quarterly']
    
    # Helper to extract last valid value from a DataFrame (annual or quarterly)
    def get_last_valid(df, field_names):
        if df is None or df.empty:
            return np.nan
        if isinstance(field_names, str):
            field_names = [field_names]
        for name in field_names:
            if name in df.index:
                series = df.loc[name].dropna()
                if not series.empty:
                    return float(series.iloc[-1])
        return np.nan
    
    # Helper to get series for trends
    def get_series_safe(df, field_names, max_years=10):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if not isinstance(field_names, list):
            field_names = [field_names]
        for name in field_names:
            if name in df.index:
                series = df.loc[name].iloc[-max_years:]
                series = pd.to_numeric(series, errors='coerce')
                return series
        return pd.Series(dtype=float)
    
    # ----- Income Statement: try annual first, then quarterly -----
    inc_annual = annual['income']
    inc_quarterly = quarterly['income']
    
    def get_income_value(field_names):
        val = get_last_valid(inc_annual, field_names)
        if not np.isnan(val):
            return val
        return get_last_valid(inc_quarterly, field_names)
    
    latest_rev = get_income_value(['Total Revenue', 'Revenue', 'Revenues', 'Sales'])
    latest_gp = get_income_value(['Gross Profit'])
    if np.isnan(latest_gp):
        cost_rev = get_income_value(['Cost Of Revenue', 'Cost of Goods Sold'])
        if not np.isnan(latest_rev) and not np.isnan(cost_rev):
            latest_gp = latest_rev - cost_rev
    latest_op = get_income_value(['Operating Income', 'EBIT', 'Operating Profit'])
    latest_ni = get_income_value(['Net Income', 'Net Income Common Stockholders', 'Net Profit'])
    latest_interest = get_income_value(['Interest Expense', 'Interest Paid'])
    
    # SG&A, R&D, Depreciation – with AAPL-specific exact names
    latest_sganda = get_income_value([
        'Selling General & Administrative', 
        'SG&A', 
        'Selling, General & Administrative', 
        'Selling, General & Administrative Expense',
        'Selling General And Administration'
    ])
    latest_rnd = get_income_value([
        'Research & Development', 
        'R&D', 
        'Research and Development',
        'Research And Development'
    ])
    latest_dep = get_income_value([
        'Depreciation & Amortization', 
        'Depreciation', 
        'Depreciation and Amortization',
        'Reconciled Depreciation'
    ])
    
    # For EPS trend, use net income series (prefer annual)
    net_income_series = get_series_safe(inc_annual, ['Net Income', 'Net Income Common Stockholders', 'Net Profit'])
    if net_income_series.empty:
        net_income_series = get_series_safe(inc_quarterly, ['Net Income', 'Net Income Common Stockholders', 'Net Profit'])
    
    # ----- Quarterly EPS: fetch directly from yfinance (multiple methods) -----
    quarterly_eps = pd.Series(dtype=float)
    ticker_obj = yf.Ticker(ticker)
    
    # Method 1: quarterly_earnings
    try:
        earnings = ticker_obj.quarterly_earnings
        if earnings is not None and not earnings.empty:
            eps_col = next((col for col in ['eps', 'EPS', 'eps_actual', 'reported_eps'] if col in earnings.columns), None)
            if eps_col:
                quarterly_eps = earnings[eps_col].sort_index().iloc[-8:]
    except:
        pass
    
    # Method 2: earnings (sometimes quarterly)
    if quarterly_eps.empty:
        try:
            earnings = ticker_obj.earnings
            if earnings is not None and not earnings.empty:
                eps_col = next((col for col in ['eps', 'EPS', 'eps_actual', 'reported_eps'] if col in earnings.columns), None)
                if eps_col:
                    quarterly_eps = earnings[eps_col].sort_index().iloc[-8:]
        except:
            pass
    
    # Method 3: quarterly_income_stmt – look for EPS row
    if quarterly_eps.empty:
        try:
            inc_q = ticker_obj.quarterly_income_stmt
            if inc_q is not None and not inc_q.empty:
                eps_rows = [row for row in inc_q.index if 'Basic EPS' in row or 'Diluted EPS' in row or 'Earnings Per Share' in row]
                if eps_rows:
                    eps_series = inc_q.loc[eps_rows[0]].dropna()
                    eps_series = eps_series.sort_index()
                    quarterly_eps = eps_series.iloc[-8:]
                else:
                    # Fallback: compute EPS from Net Income and shares outstanding
                    net_income = inc_q.loc['Net Income'] if 'Net Income' in inc_q.index else pd.Series(dtype=float)
                    if not net_income.empty:
                        # Try to get shares from balance sheet
                        bal_q = ticker_obj.quarterly_balance_sheet
                        shares = bal_q.loc['Common Stock Shares Outstanding'] if bal_q is not None and 'Common Stock Shares Outstanding' in bal_q.index else pd.Series(dtype=float)
                        if not shares.empty:
                            common_idx = net_income.index.intersection(shares.index)
                            if len(common_idx) > 0:
                                eps_computed = net_income[common_idx] / shares[common_idx]
                                quarterly_eps = eps_computed.iloc[-8:]
        except Exception as e:
            st.warning(f"Could not extract quarterly EPS via income statement: {e}")
    
    if quarterly_eps.empty:
        st.warning(f"No quarterly EPS data available for {ticker} after all methods.")
    
    # ----- Earnings surprise (beat / miss) using get_earnings_dates -----
    earnings_surprise = pd.DataFrame()
    try:
        earnings_dates = ticker_obj.get_earnings_dates(limit=8)
        if earnings_dates is not None and not earnings_dates.empty:
            # Detect actual column names (case‑insensitive)
            cols = {col.lower(): col for col in earnings_dates.columns}
            eps_est_col = cols.get('eps estimate', None)
            eps_act_col = cols.get('reported eps', None) or cols.get('eps actual', None)
            surprise_col = cols.get('surprise(%)', None)
            if eps_est_col and eps_act_col and surprise_col:
                earnings_dates['eps_estimate'] = earnings_dates[eps_est_col]
                earnings_dates['eps_actual'] = earnings_dates[eps_act_col]
                earnings_dates['surprise_pct'] = earnings_dates[surprise_col]
                earnings_dates = earnings_dates[['eps_actual', 'eps_estimate', 'surprise_pct']]
                earnings_dates.index = pd.to_datetime(earnings_dates.index)
                earnings_dates = earnings_dates.sort_index()  # oldest first
                earnings_surprise = earnings_dates.iloc[-8:]
            else:
                st.warning(f"Could not find required columns in earnings data for {ticker}. Found: {list(earnings_dates.columns)}")
        else:
            st.info(f"No earnings dates data for {ticker}")
    except Exception as e:
        st.warning(f"Could not fetch earnings surprise data for {ticker}: {e}")
    
    # Operating profit consistency: take last 3 non‑null values
    op_series = get_series_safe(inc_annual, ['Operating Income', 'EBIT', 'Operating Profit']).dropna()
    if len(op_series) < 3:
        op_series = get_series_safe(inc_quarterly, ['Operating Income', 'EBIT', 'Operating Profit']).dropna()
    oper_consistent = (op_series.iloc[-3:] > 0).all() if len(op_series) >= 3 else False
    
    # ----- Balance Sheet: try annual first, then quarterly -----
    bal_annual = annual['balance']
    bal_quarterly = quarterly['balance']
    
    def get_balance_value(field_names):
        val = get_last_valid(bal_annual, field_names)
        if not np.isnan(val):
            return val
        return get_last_valid(bal_quarterly, field_names)
    
    total_assets = get_balance_value(['Total Assets', 'Assets'])
    total_liabilities = get_balance_value(['Total Liabilities Net Minority Interest', 'Total Liabilities', 'Liabilities'])
    shareholders_equity = get_balance_value(['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Total Equity'])
    current_assets = get_balance_value(['Current Assets', 'Total Current Assets'])
    current_liabilities = get_balance_value(['Current Liabilities', 'Total Current Liabilities'])
    long_term_debt = get_balance_value(['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'])
    inventory = get_balance_value(['Inventory', 'Total Inventory'])
    receivables = get_balance_value(['Receivables', 'Accounts Receivable', 'Gross Accounts Receivable'])
    
    # Shares outstanding – ensure full length series (for annual trend)
    shares_series_raw = get_series_safe(bal_annual, ['Common Stock Shares Outstanding'])
    if shares_series_raw.empty:
        shares_series_raw = get_series_safe(bal_quarterly, ['Common Stock Shares Outstanding'])
    
    if not shares_series_raw.empty:
        latest_shares = shares_series_raw.dropna().iloc[-1]
        shares_series = pd.Series(latest_shares, index=net_income_series.index)
    else:
        try:
            info = yf.Ticker(ticker).info
            shares = info.get('sharesOutstanding')
            if shares:
                shares_series = pd.Series(shares, index=net_income_series.index)
            else:
                shares_series = pd.Series(dtype=float)
        except:
            shares_series = pd.Series(dtype=float)
    
    # ----- Cash Flow: get single latest CapEx value and series -----
    cf_annual = annual['cashflow']
    cf_quarterly = quarterly['cashflow']
    
    def get_capex_value():
        val = get_last_valid(cf_annual, ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of Property Plant Equipment'])
        if not np.isnan(val):
            return abs(val)
        val = get_last_valid(cf_quarterly, ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of Property Plant Equipment'])
        return abs(val) if not np.isnan(val) else np.nan
    
    capex_latest = get_capex_value()
    capex_series = get_series_safe(cf_annual, ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of Property Plant Equipment'])
    if capex_series.empty:
        capex_series = get_series_safe(cf_quarterly, ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of Property Plant Equipment'])
    capex_abs = capex_series.abs() if not capex_series.empty else pd.Series(dtype=float)
    
    # ----- Compute ratios -----
    gpm = (latest_gp / latest_rev * 100) if latest_rev not in [0, np.nan] else np.nan
    npm = (latest_ni / latest_rev * 100) if latest_rev not in [0, np.nan] else np.nan
    sganda_ratio = (latest_sganda / latest_gp * 100) if latest_gp not in [0, np.nan] else np.nan
    rnd_ratio = (latest_rnd / latest_gp * 100) if latest_gp not in [0, np.nan] else np.nan
    
    # Guard for dep_high
    if not np.isnan(latest_gp) and latest_gp != 0:
        dep_high = (latest_dep / latest_gp > 0.20)
    else:
        dep_high = False
    
    interest_coverage = latest_op / latest_interest if latest_interest not in [0, np.nan] else np.nan
    current_ratio = current_assets / current_liabilities if current_liabilities not in [0, np.nan] else np.nan
    debt_to_equity = total_liabilities / shareholders_equity if shareholders_equity not in [0, np.nan] else np.nan
    roe = (latest_ni / shareholders_equity * 100) if shareholders_equity not in [0, np.nan] else np.nan
    debt_payback = long_term_debt / latest_ni if latest_ni > 0 else np.nan
    inv_rec_ratio = inventory / receivables if receivables not in [0, np.nan] else np.nan
    roa = (latest_ni / total_assets * 100) if total_assets not in [0, np.nan] else np.nan
    capex_annual_ratio = (capex_latest / latest_ni * 100) if not np.isnan(capex_latest) and latest_ni not in [0, np.nan] else np.nan
    
    # 10-year CapEx efficiency
    capex_10y_ratio = np.nan
    if len(capex_abs) >= 5 and len(net_income_series) >= 5 and len(shares_series) >= 5:
        min_len = min(len(capex_abs), len(net_income_series), len(shares_series))
        capex_ps = (capex_abs.iloc[-min_len:] / shares_series.iloc[-min_len:]).sum()
        eps_total = (net_income_series.iloc[-min_len:] / shares_series.iloc[-min_len:]).sum()
        if eps_total not in [0, np.nan]:
            capex_10y_ratio = (capex_ps / eps_total) * 100
    
    eps_trend = assess_eps_trend_annual(net_income_series, shares_series)
    
    metrics = {
        'gpm': gpm,
        'oper_consistent': oper_consistent,
        'npm': npm,
        'sganda_ratio': sganda_ratio,
        'rnd_ratio': rnd_ratio,
        'dep_high': dep_high,
        'interest_coverage': interest_coverage,
        'eps_trend': eps_trend,
        'current_ratio': current_ratio,
        'debt_to_equity': debt_to_equity,
        'roe': roe,
        'debt_payback': debt_payback,
        'inv_rec_ratio': inv_rec_ratio,
        'roa': roa,
        'capex_annual': capex_annual_ratio,
        'capex_10y': capex_10y_ratio,
        'revenue_series': pd.Series(),
        'net_income_series': net_income_series,
        'quarterly_eps': quarterly_eps,                 # actual EPS series
        'earnings_surprise': earnings_surprise,        # NEW: DataFrame with estimates & surprises
    }
    return metrics

# ------------------------------------------------------------
# Scoring (unchanged, but uses metrics with guarded dep_high)
# ------------------------------------------------------------
def compute_tally_and_grade(metrics):
    strong = 0
    moderate = 0
    weak = 0
    details = {}
    
    # GPM
    gpm = metrics['gpm']
    if not np.isnan(gpm):
        if gpm > 40:
            strong += 1
            details['Gross Profit Margin'] = f'✅ Strong ({gpm:.1f}%)'
        elif 20 <= gpm <= 40:
            moderate += 1
            details['Gross Profit Margin'] = f'⚠️ Moderate ({gpm:.1f}%)'
        else:
            weak += 1
            details['Gross Profit Margin'] = f'❌ Weak ({gpm:.1f}%)'
    else:
        details['Gross Profit Margin'] = 'N/A'
    
    # Operating Profit
    if metrics['oper_consistent']:
        strong += 1
        details['Operating Profit'] = '✅ Consistent profit'
    else:
        weak += 1
        details['Operating Profit'] = '❌ Loss in recent years'
    
    # Net Profit Margin
    npm = metrics['npm']
    if not np.isnan(npm):
        if npm >= 20:
            strong += 1
            details['Net Profit Margin'] = f'✅ Great (≥20%)'
        elif 10 <= npm < 20:
            moderate += 1
            details['Net Profit Margin'] = f'⚠️ Good (10-20%)'
        else:
            weak += 1
            details['Net Profit Margin'] = f'❌ Low (<10%)'
    else:
        details['Net Profit Margin'] = 'N/A'
    
    # SG&A Efficiency
    sg = metrics['sganda_ratio']
    if not np.isnan(sg):
        if sg < 30:
            strong += 1
            details['SG&A / Gross Profit'] = f'✅ Excellent ({sg:.1f}%)'
        elif 30 <= sg <= 80:
            moderate += 1
            details['SG&A / Gross Profit'] = f'⚠️ Acceptable ({sg:.1f}%)'
        else:
            weak += 1
            details['SG&A / Gross Profit'] = f'❌ Concerning ({sg:.1f}%)'
    else:
        details['SG&A / Gross Profit'] = 'N/A'
    
    details['R&D / Gross Profit'] = f"{metrics['rnd_ratio']:.1f}%" if not np.isnan(metrics['rnd_ratio']) else "N/A"
    
    # Depreciation
    if metrics['dep_high']:
        weak += 1
        details['Depreciation'] = '❌ High (capital intensive)'
    else:
        strong += 1
        details['Depreciation'] = '✅ Low (not capital intensive)'
    
    # Interest Coverage
    ic = metrics['interest_coverage']
    if not np.isnan(ic):
        if ic > 5:
            strong += 1
            details['Interest Coverage'] = f'✅ Strong ({ic:.1f}x)'
        elif ic >= 3:
            moderate += 1
            details['Interest Coverage'] = f'⚠️ Adequate ({ic:.1f}x)'
        else:
            weak += 1
            details['Interest Coverage'] = f'❌ Risky ({ic:.1f}x)'
    else:
        details['Interest Coverage'] = 'N/A'
    
    # EPS Trend
    eps_t = metrics['eps_trend']
    if eps_t == 'strong':
        strong += 1
        details['EPS Trend (7y)'] = '✅ Strong upward trend'
    elif eps_t == 'moderate':
        moderate += 1
        details['EPS Trend (7y)'] = '⚠️ Moderate / mixed'
    elif eps_t == 'weak':
        weak += 1
        details['EPS Trend (7y)'] = '❌ Weak / boom-bust'
    else:
        details['EPS Trend (7y)'] = 'Insufficient data'
    
    # Current Ratio
    cr = metrics['current_ratio']
    if not np.isnan(cr):
        if cr > 1:
            strong += 1
            details['Current Ratio'] = f'✅ >1 ({cr:.2f})'
        else:
            weak += 1
            details['Current Ratio'] = f'❌ <1 ({cr:.2f})'
    else:
        details['Current Ratio'] = 'N/A'
    
    # Debt/Equity
    de = metrics['debt_to_equity']
    if not np.isnan(de):
        if de < 0.8:
            strong += 1
            details['Debt/Equity'] = f'✅ Good ({de:.2f})'
        elif de < 1.5:
            moderate += 1
            details['Debt/Equity'] = f'⚠️ Moderate ({de:.2f})'
        else:
            weak += 1
            details['Debt/Equity'] = f'❌ High ({de:.2f})'
    else:
        details['Debt/Equity'] = 'N/A'
    
    # ROE
    roe = metrics['roe']
    if not np.isnan(roe):
        if roe >= 20:
            strong += 1
            details['ROE'] = f'✅ Great ({roe:.1f}%)'
        else:
            moderate += 1
            details['ROE'] = f'⚠️ Below 20% ({roe:.1f}%)'
    else:
        details['ROE'] = 'N/A'
    
    # Debt Payback
    dp = metrics['debt_payback']
    if not np.isnan(dp):
        if dp <= 4:
            strong += 1
            details['Debt Payback (years)'] = f'✅ ≤4 years ({dp:.1f})'
        elif dp <= 7:
            moderate += 1
            details['Debt Payback (years)'] = f'⚠️ 4-7 years ({dp:.1f})'
        else:
            weak += 1
            details['Debt Payback (years)'] = f'❌ >7 years ({dp:.1f})'
    else:
        details['Debt Payback (years)'] = 'N/A'
    
    # Inventory/Receivables
    inv_rec = metrics['inv_rec_ratio']
    details['Inventory/Receivables'] = f"{inv_rec:.2f}" if not np.isnan(inv_rec) else "N/A"
    
    # ROA
    roa = metrics['roa']
    if not np.isnan(roa):
        if roa > 10:
            strong += 1
            details['ROA'] = f'✅ Strong ({roa:.1f}%) – LBO resistant'
        elif roa > 5:
            moderate += 1
            details['ROA'] = f'⚠️ Moderate ({roa:.1f}%)'
        else:
            weak += 1
            details['ROA'] = f'❌ Low ({roa:.1f}%) – LBO vulnerable'
    else:
        details['ROA'] = 'N/A'
    
    # CapEx Annual
    ca = metrics['capex_annual']
    if not np.isnan(ca):
        if ca <= 25:
            strong += 1
            details['CapEx / Net Income (annual)'] = f'✅ Great (≤25%)'
        elif ca <= 50:
            moderate += 1
            details['CapEx / Net Income (annual)'] = f'⚠️ Good (≤50%)'
        else:
            weak += 1
            details['CapEx / Net Income (annual)'] = f'❌ Concerning (>50%)'
    else:
        details['CapEx / Net Income (annual)'] = 'N/A'
    
    # CapEx 10Y
    c10 = metrics['capex_10y']
    if not np.isnan(c10):
        if c10 <= 25:
            strong += 1
            details['CapEx / EPS (10Y)'] = f'✅ Great (≤25%)'
        elif c10 <= 50:
            moderate += 1
            details['CapEx / EPS (10Y)'] = f'⚠️ Good (≤50%)'
        else:
            weak += 1
            details['CapEx / EPS (10Y)'] = f'❌ Concerning (>50%)'
    else:
        details['CapEx / EPS (10Y)'] = 'N/A'
    
    total = strong + moderate + weak
    if total == 0:
        grade = 'Insufficient Data'
    else:
        strong_pct = strong / total
        if strong_pct >= 0.8:
            grade = 'A'
        elif strong_pct >= 0.6:
            grade = 'B'
        elif strong_pct >= 0.4:
            grade = 'C'
        elif strong_pct >= 0.2:
            grade = 'D'
        else:
            grade = 'F'
    
    return strong, moderate, weak, grade, details

# ------------------------------------------------------------
# Main public function
# ------------------------------------------------------------
@st.cache_data(ttl=86400)
def get_advanced_durability(ticker: str):
    metrics = analyze_company(ticker)
    if metrics is None:
        return None
    strong, moderate, weak, grade, details = compute_tally_and_grade(metrics)
    return {
        'grade': grade,
        'strong_count': strong,
        'moderate_count': moderate,
        'weak_count': weak,
        'details': details,
        'metrics': metrics,
        'quarterly_eps': metrics.get('quarterly_eps', pd.Series(dtype=float)),
        'earnings_surprise': metrics.get('earnings_surprise', pd.DataFrame())  # NEW
    }