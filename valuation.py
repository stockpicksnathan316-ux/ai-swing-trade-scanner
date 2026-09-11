import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sec_data import fetch_sec_financials


# ------------------- Simple DCF (original) -------------------
def get_10y_fcf(ticker):
    """Fetch Free Cash Flow data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        cashflow = stock.cashflow
        if cashflow.empty:
            return None, "No cash flow data available"
        if 'Free Cash Flow' in cashflow.index:
            fcf_series = cashflow.loc['Free Cash Flow']
        else:
            ocf = cashflow.loc['Operating Cash Flow'] if 'Operating Cash Flow' in cashflow.index else None
            capex = cashflow.loc['Capital Expenditure'] if 'Capital Expenditure' in cashflow.index else None
            if ocf is not None and capex is not None:
                fcf_series = ocf - abs(capex)
            else:
                return None, "FCF data not available for this ticker"
        fcf_series = fcf_series.astype(float).dropna()
        if len(fcf_series) < 3:
            return None, f"Only {len(fcf_series)} years of FCF data available (need at least 3)"
        return fcf_series, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


def calculate_growth_rate(fcf_series):
    """Calculate historical CAGR from FCF data."""
    if len(fcf_series) < 5:
        return 0.0, False
    fcf_series = fcf_series.astype(float)
    first = fcf_series.iloc[-1]
    last = fcf_series.iloc[0]
    years = len(fcf_series) - 1
    if first <= 0 or last <= 0 or np.isnan(first) or np.isnan(last):
        return 0.0, False
    try:
        growth = (last / first) ** (1 / years) - 1
    except Exception:
        return 0.0, False
    growth = min(growth, 0.30)
    growth = max(growth, -0.10)
    return growth, True


def calculate_intrinsic_value(ticker, projection_years=10, discount_rate=0.15, margin_of_safety=0.30):
    """Simple DCF using historical FCF growth (original method)."""
    fcf_series, error = get_10y_fcf(ticker)
    if error:
        return {'error': error}
    if len(fcf_series) < 5:
        growth_rate = 0.0
        data_warning = f"⚠️ Only {len(fcf_series)} years of FCF data available. Using 0% growth rate (conservative)."
    else:
        growth_rate, reliable = calculate_growth_rate(fcf_series)
        if not reliable or np.isnan(growth_rate):
            growth_rate = 0.0
        data_warning = None
    latest_fcf = fcf_series.iloc[0]
    if np.isnan(latest_fcf) or latest_fcf <= 0:
        return {'error': 'Latest FCF is zero or negative – cannot project'}
    stock = yf.Ticker(ticker)
    info = stock.info
    shares_outstanding = info.get('sharesOutstanding', 1)
    current_price = info.get('regularMarketPrice', 0)
    total_pv = 0
    projected_fcf_list = []
    pv_list = []
    for n in range(1, projection_years + 1):
        projected_fcf = latest_fcf * (1 + growth_rate) ** n
        pv = projected_fcf / (1 + discount_rate) ** n
        total_pv += pv
        projected_fcf_list.append(projected_fcf)
        pv_list.append(pv)
    perpetual_growth = 0.02
    terminal_fcf = latest_fcf * (1 + growth_rate) ** (projection_years + 1)
    terminal_value = terminal_fcf / (discount_rate - perpetual_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** projection_years
    total_value = total_pv + terminal_pv
    intrinsic_value_per_share = total_value / shares_outstanding
    if np.isnan(intrinsic_value_per_share) or intrinsic_value_per_share <= 0:
        return {'error': 'Intrinsic value calculation resulted in invalid number'}
    max_buy_price = intrinsic_value_per_share * (1 - margin_of_safety)
    if current_price <= max_buy_price:
        verdict = "✅ Buy (Undervalued)"
        verdict_color = "green"
    elif current_price <= intrinsic_value_per_share:
        verdict = "⏳ Hold / Watch (Fairly Valued)"
        verdict_color = "yellow"
    else:
        verdict = "❌ Sell / Overvalued"
        verdict_color = "red"
    discount_pct = (intrinsic_value_per_share - current_price) / intrinsic_value_per_share * 100
    return {
        'intrinsic_value': intrinsic_value_per_share,
        'max_buy_price': max_buy_price,
        'current_price': current_price,
        'margin_of_safety': margin_of_safety,
        'discount_pct': discount_pct,
        'verdict': verdict,
        'verdict_color': verdict_color,
        'growth_rate': growth_rate,
        'latest_fcf': latest_fcf,
        'terminal_value': terminal_value,
        'terminal_pv': terminal_pv,
        'total_pv': total_pv,
        'shares_outstanding': shares_outstanding,
        'projected_fcf': projected_fcf_list,
        'pv_list': pv_list,
        'error': None,
        'data_warning': data_warning,
        'method': 'Simple DCF (FCF-based)'
    }


# ------------------- Revenue growth helpers -------------------
def ttm_cagr(income_quarterly, years=5):
    """Compute revenue CAGR using TTM sums from quarterly data."""
    if income_quarterly is None or income_quarterly.empty:
        return None
    rev_q = None
    for name in ['Total Revenue', 'Revenue', 'TotalRevenue']:
        if name in income_quarterly.index:
            rev_q = income_quarterly.loc[name]
            break
    if rev_q is None:
        return None
    rev_q = rev_q.dropna()
    needed = 4 * (years + 1)
    if len(rev_q) < needed:
        return None
    try:
        ttm_now = rev_q.iloc[:4].sum()
        ttm_old = rev_q.iloc[4*years : 4*years + 4].sum()
        if ttm_old <= 0 or ttm_now <= 0:
            return None
        return (ttm_now / ttm_old) ** (1 / years) - 1
    except Exception:
        return None


def annual_cagr(revenue_series, years=5):
    """Annual revenue CAGR with relaxed minimum (3 years)."""
    if revenue_series is None or len(revenue_series) < 3:
        return None
    ser = revenue_series.iloc[:min(years, len(revenue_series))].dropna()
    if len(ser) < 3:
        return None
    first = ser.iloc[-1]
    last = ser.iloc[0]
    if first <= 0 or last <= 0:
        return None
    try:
        return (last / first) ** (1 / (len(ser) - 1)) - 1
    except Exception:
        return None


def avg_yoy_growth(revenue_series, years=4):
    """Average of the last `years` YoY growth rates."""
    if revenue_series is None or len(revenue_series) < 2:
        return None
    ser = revenue_series.iloc[:min(years + 1, len(revenue_series))].dropna()
    if len(ser) < 2:
        return None
    growths = []
    for i in range(len(ser) - 1):
        latest = ser.iloc[i]
        prior = ser.iloc[i + 1]
        if prior > 0 and latest > 0:
            growths.append((latest - prior) / prior)
    if not growths:
        return None
    return sum(growths) / len(growths)


# ------------------- SEC-backed metrics -------------------
def get_financial_metrics_sec(ticker):
    """
    Fetch financial metrics using SEC EDGAR data (10+ years of history).
    Returns the same dict shape as get_financial_metrics(), or None if
    SEC data isn't available for this ticker.
    """
    try:
        sec = fetch_sec_financials(ticker)
        if not sec or sec.get("revenue") is None:
            return None

        # Get spot info (price, shares outstanding, forward estimates) from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info

        # ----- Revenue growth: 5-year CAGR from SEC annual data -----
        rev = sec["revenue"].dropna()
        rev = rev[rev > 0]
        rev_growth_hist = None
        if len(rev) >= 6:
            rev_growth_hist = (rev.iloc[0] / rev.iloc[5]) ** (1 / 5) - 1
        elif len(rev) >= 4:
            rev_growth_hist = (rev.iloc[0] / rev.iloc[3]) ** (1 / 3) - 1
        elif len(rev) >= 2:
            n = len(rev) - 1
            rev_growth_hist = (rev.iloc[0] / rev.iloc[-1]) ** (1 / n) - 1

        # ----- Latest values for reference -----
        latest_revenue = rev.iloc[0] if len(rev) > 0 else None
        ni = sec.get("net_income")
        latest_net_income = ni.iloc[0] if ni is not None and len(ni) > 0 else None
        fcf = sec.get("fcf")
        latest_fcf = fcf.iloc[0] if fcf is not None and len(fcf) > 0 else None

        # ----- 5-year average Net Margin & Cash Conversion -----
        net_margin_hist = None
        cash_conversion_hist = None
        
        if ni is not None and len(ni) > 0:
            # Align revenue and net income on common dates
            common = rev.index.intersection(ni.index)
            if len(common) >= 3:
                # Take the 5 most recent years
                common = sorted(common, reverse=True)[:5]
                margins = [ni.loc[d] / rev.loc[d] for d in common if rev.loc[d] > 0]
                if margins:
                    net_margin_hist = sum(margins) / len(margins)

        if fcf is not None and ni is not None and len(fcf) > 0 and len(ni) > 0:
            # Align FCF and net income on common dates
            common = fcf.index.intersection(ni.index)
            if len(common) >= 3:
                common = sorted(common, reverse=True)[:5]
                conversions = [fcf.loc[d] / ni.loc[d] for d in common if ni.loc[d] > 0]
                if conversions:
                    cash_conversion_hist = sum(conversions) / len(conversions)

        # ----- Net cash from yfinance balance sheet -----
        # CRITICAL: initialize these before the try, so they're always defined
        short_term_debt = 0
        long_term_debt = 0
        net_cash = 0
        try:
            balance = stock.balance_sheet
            cash = balance.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance.index else 0
            short_term_debt = balance.loc['Short Term Debt'].iloc[0] if 'Short Term Debt' in balance.index else 0
            long_term_debt = balance.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance.index else 0
            net_cash = cash - (short_term_debt + long_term_debt)
        except Exception:
            net_cash = 0

        # ----- Shares, price, forward estimates -----
        shares = info.get('sharesOutstanding', 1)
        if not shares or shares <= 0:
            shares = info.get('floatShares') or 1
        current_price = info.get('regularMarketPrice', 0)

        forward_rev_growth = info.get('revenueGrowth')
        forward_net_margin = info.get('profitMargins')
        forward_earnings_growth = info.get('earningsGrowth')
        current_ps_ratio = info.get('priceToSalesTrailing12Months')

        # ----- Simple WACC -----
        wacc = 0.088
        try:
            beta = info.get('beta', 1.0)
            risk_free = 0.03
            try:
                tnx = yf.Ticker('^TNX')
                hist = tnx.history(period='1d')
                if not hist.empty:
                    risk_free = hist['Close'].iloc[-1] / 100.0
            except Exception:
                pass
            cost_of_equity = risk_free + beta * 0.055
            market_cap = shares * current_price
            cost_of_debt = 0.05
            tax_rate = 0.21
            total_debt = short_term_debt + long_term_debt
            total_capital = market_cap + total_debt
            if total_capital > 0:
                wacc = (market_cap / total_capital) * cost_of_equity + \
                       (total_debt / total_capital) * cost_of_debt * (1 - tax_rate)
        except Exception:
            pass
        if wacc <= 0 or wacc > 0.5:
            wacc = 0.088

        # Cap growth
        if rev_growth_hist is not None:
            rev_growth_hist = max(-0.5, min(1.0, rev_growth_hist))

        return {
            'latest_revenue': latest_revenue,
            'latest_net_income': latest_net_income,
            'latest_fcf': latest_fcf,
            'shares_outstanding': shares,
            'current_price': current_price,
            'net_cash': net_cash,
            'revenue_growth_hist': rev_growth_hist,
            'net_margin_hist': net_margin_hist,
            'cash_conversion_hist': cash_conversion_hist,
            'company_name': sec.get('entity_name', info.get('longName', ticker)),
            'forward_revenue_growth': forward_rev_growth,
            'forward_net_margin': forward_net_margin,
            'forward_earnings_growth': forward_earnings_growth,
            'current_ps_ratio': current_ps_ratio,
            'historical_ps_avg': None,
            'wacc': wacc,
            'data_source': 'SEC EDGAR',
        }
    except Exception:
        return None


# ------------------- Advanced DCF (Alpha Spread style) -------------------
def get_financial_metrics(ticker, prefer_sec=True):
    """
    Fetch key financial metrics. Tries SEC EDGAR first (10+ years of clean
    annual data), falls back to yfinance if SEC is unavailable.
    """
    # Try SEC first
    if prefer_sec:
        sec_metrics = get_financial_metrics_sec(ticker)
        if sec_metrics is not None:
            return sec_metrics

    # Fallback: yfinance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        income_annual = stock.financials
        income_quarterly = stock.quarterly_income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow

        def get_row_value(df, possible_names):
            if df is None or df.empty:
                return None
            for name in possible_names:
                if name in df.index:
                    return df.loc[name]
            return None

        # ----- TTM Revenue -----
        latest_revenue = None
        if income_quarterly is not None and not income_quarterly.empty:
            try:
                rev_q = get_row_value(income_quarterly, ['Total Revenue', 'Revenue', 'TotalRevenue'])
                if rev_q is not None and len(rev_q) >= 4:
                    latest_revenue = rev_q.iloc[:4].sum()
            except Exception:
                pass
        if latest_revenue is None or latest_revenue <= 0:
            try:
                rev_annual = get_row_value(income_annual, ['Total Revenue', 'Revenue', 'TotalRevenue'])
                latest_revenue = rev_annual.iloc[0] if rev_annual is not None else None
            except Exception:
                pass
        if latest_revenue is None or latest_revenue <= 0:
            latest_revenue = info.get('totalRevenue') or info.get('revenue') or 0

        # ----- TTM Net Income -----
        latest_net_income = None
        if income_quarterly is not None and not income_quarterly.empty:
            try:
                ni_q = get_row_value(income_quarterly, ['Net Income', 'NetIncome'])
                if ni_q is not None and len(ni_q) >= 4:
                    latest_net_income = ni_q.iloc[:4].sum()
            except Exception:
                pass
        if latest_net_income is None or latest_net_income <= 0:
            try:
                ni_annual = get_row_value(income_annual, ['Net Income', 'NetIncome'])
                latest_net_income = ni_annual.iloc[0] if ni_annual is not None else None
            except Exception:
                pass
        if latest_net_income is None or latest_net_income <= 0:
            latest_net_income = info.get('netIncomeToCommon') or 0

        # ----- Shares -----
        shares = info.get('sharesOutstanding', 1)
        if shares is None or shares <= 0:
            shares = info.get('floatShares') or 1

        # ----- Net cash from yfinance balance sheet -----
        short_term_debt = 0
        long_term_debt = 0
        net_cash = 0
        try:
            balance = stock.balance_sheet

            def _row(names, default=0):
                for n in names:
                    if n in balance.index:
                        try:
                            val = balance.loc[n].iloc[0]
                            if val is not None and not np.isnan(val):
                                return float(val)
                        except Exception:
                            pass
                return default

            # Cash + all liquid investments (short and long term)
            cash = _row(['Cash And Cash Equivalents', 'Cash'])
            st_inv = _row(['Other Short Term Investments', 'Short Term Investments'])
            lt_inv = _row(['Investments And Advances', 'Long Term Investments',
                           'InvestmentinFinancialAssets'])

            # All debt (yfinance uses "Current Debt" not "Short Term Debt")
            st_debt = _row(['Current Debt', 'Short Term Debt', 'Current Debt And Capital Lease Obligation'])
            lt_debt = _row(['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'])

            total_cash_like = cash + st_inv + lt_inv
            total_debt = st_debt + lt_debt
            net_cash = total_cash_like - total_debt

            short_term_debt = st_debt   # for downstream WACC calc
            long_term_debt = lt_debt
        except Exception:
            net_cash = 0

        # ----- Current Price -----
        current_price = info.get('regularMarketPrice', 0)

        # ----- Historical series -----
        revenue_series = get_row_value(income_annual, ['Total Revenue', 'Revenue', 'TotalRevenue'])
        net_income_series = get_row_value(income_annual, ['Net Income', 'NetIncome'])
        fcf_series = get_row_value(cashflow, ['Free Cash Flow', 'FreeCashFlow'])

        latest_revenue_annual = revenue_series.iloc[0] if revenue_series is not None else None
        latest_net_income_annual = net_income_series.iloc[0] if net_income_series is not None else None
        latest_fcf_annual = fcf_series.iloc[0] if fcf_series is not None else None

        # Net margin
        if latest_revenue_annual and latest_net_income_annual and latest_revenue_annual > 0:
            net_margin_hist = latest_net_income_annual / latest_revenue_annual
        else:
            net_margin_hist = (latest_net_income / latest_revenue) if latest_revenue and latest_revenue > 0 else None

        # Cash conversion
        if latest_fcf_annual and latest_net_income_annual and latest_net_income_annual > 0:
            cash_conversion_hist = latest_fcf_annual / latest_net_income_annual
        else:
            cash_conversion_hist = (latest_fcf_annual / latest_net_income) if latest_fcf_annual and latest_net_income and latest_net_income > 0 else None

        # ----- Revenue growth cascade -----
        rev_growth_hist = ttm_cagr(income_quarterly, years=5)
        if rev_growth_hist is None:
            rev_growth_hist = annual_cagr(revenue_series, years=5)
        if rev_growth_hist is None:
            rev_growth_hist = annual_cagr(revenue_series, years=3)
        if rev_growth_hist is None:
            rev_growth_hist = avg_yoy_growth(revenue_series, years=4)

        if rev_growth_hist is not None:
            rev_growth_hist = max(-0.5, min(1.0, rev_growth_hist))

        # ----- P/S ratio -----
        current_ps_ratio = info.get('priceToSalesTrailing12Months')
        hist_ps_avg = None

        # ----- Simple WACC -----
        wacc = None
        try:
            beta = info.get('beta', 1.0)
            risk_free = 0.03
            try:
                tnx = yf.Ticker('^TNX')
                hist = tnx.history(period='1d')
                if not hist.empty:
                    risk_free = hist['Close'].iloc[-1] / 100.0
            except Exception:
                pass
            market_risk_premium = 0.055
            cost_of_equity = risk_free + beta * market_risk_premium

            interest_expense = None
            if income_annual is not None and 'Interest Expense' in income_annual.index:
                interest_expense = income_annual.loc['Interest Expense'].iloc[0]
            total_debt = short_term_debt + long_term_debt
            if total_debt > 0 and interest_expense is not None and interest_expense != 0:
                cost_of_debt = abs(interest_expense) / total_debt
            else:
                cost_of_debt = 0.05

            tax_rate = 0.21
            if income_annual is not None and 'Tax Provision' in income_annual.index and 'Pretax Income' in income_annual.index:
                tax_prov = income_annual.loc['Tax Provision'].iloc[0]
                pretax = income_annual.loc['Pretax Income'].iloc[0]
                if pretax > 0:
                    tax_rate = tax_prov / pretax

            market_cap = shares * current_price if shares and current_price else 0
            total_capital = market_cap + total_debt
            if total_capital > 0:
                wacc = (market_cap / total_capital) * cost_of_equity + (total_debt / total_capital) * cost_of_debt * (1 - tax_rate)
            else:
                wacc = cost_of_equity
        except Exception:
            wacc = None

        if wacc is None or wacc <= 0 or wacc > 0.5:
            wacc = 0.088

        # ----- Forward estimates -----
        forward_rev_growth = info.get('revenueGrowth')
        forward_net_margin = info.get('profitMargins')
        forward_earnings_growth = info.get('earningsGrowth')

        return {
            'latest_revenue': latest_revenue,
            'latest_net_income': latest_net_income,
            'latest_fcf': latest_fcf_annual if latest_fcf_annual is not None else None,
            'shares_outstanding': shares,
            'current_price': current_price,
            'net_cash': net_cash,
            'revenue_growth_hist': rev_growth_hist,
            'net_margin_hist': net_margin_hist,
            'cash_conversion_hist': cash_conversion_hist,
            'company_name': info.get('longName', ticker),
            'forward_revenue_growth': forward_rev_growth,
            'forward_net_margin': forward_net_margin,
            'forward_earnings_growth': forward_earnings_growth,
            'current_ps_ratio': current_ps_ratio,
            'historical_ps_avg': hist_ps_avg,
            'wacc': wacc,
            'data_source': 'yfinance (fallback)',
        }
    except Exception as e:
        return {'error': str(e)}


def calculate_intrinsic_value_advanced(
    ticker,
    revenue_growth=0.096,
    net_margin=0.262,
    cash_conversion=1.01,
    discount_rate=0.088,
    exit_multiple=5.7,
    forecast_years=5,
    margin_of_safety=0.30
):
    """
    Advanced DCF model following Alpha Spread's methodology.
    """
    metrics = get_financial_metrics(ticker)
    if 'error' in metrics:
        return {'error': metrics['error']}

    latest_revenue = metrics['latest_revenue']
    shares = metrics['shares_outstanding']
    current_price = metrics['current_price']
    net_cash = metrics['net_cash']

    if latest_revenue is None or latest_revenue <= 0:
        return {'error': 'No revenue data available for this ticker'}

    if shares is None or shares <= 0:
        return {'error': 'Shares outstanding data not available'}

    # Project revenue
    revenue_projections = [latest_revenue * (1 + revenue_growth) ** (i + 1) for i in range(forecast_years)]
    net_income_projections = [rev * net_margin for rev in revenue_projections]
    fcfe_projections = [ni * cash_conversion for ni in net_income_projections]

    # Discount FCFE
    pv_fcfe = []
    for i, fcfe in enumerate(fcfe_projections):
        pv = fcfe / (1 + discount_rate) ** (i + 1)
        pv_fcfe.append(pv)

    total_pv_fcfe = sum(pv_fcfe)

    # Terminal value
    terminal_revenue = latest_revenue * (1 + revenue_growth) ** (forecast_years + 1)
    terminal_value = terminal_revenue * exit_multiple
    pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

    # Enterprise value
    enterprise_value = total_pv_fcfe + pv_terminal
    equity_value = enterprise_value + net_cash
    intrinsic_value_per_share = equity_value / shares

    if np.isnan(intrinsic_value_per_share) or intrinsic_value_per_share <= 0:
        return {'error': 'Intrinsic value calculation resulted in invalid number'}

    max_buy_price = intrinsic_value_per_share * (1 - margin_of_safety)

    if current_price <= max_buy_price:
        verdict = "✅ Buy (Undervalued)"
        verdict_color = "green"
    elif current_price <= intrinsic_value_per_share:
        verdict = "⏳ Hold / Watch (Fairly Valued)"
        verdict_color = "yellow"
    else:
        verdict = "❌ Sell / Overvalued"
        verdict_color = "red"

    discount_pct = (intrinsic_value_per_share - current_price) / intrinsic_value_per_share * 100

    return {
        'intrinsic_value': intrinsic_value_per_share,
        'max_buy_price': max_buy_price,
        'current_price': current_price,
        'margin_of_safety': margin_of_safety,
        'discount_pct': discount_pct,
        'verdict': verdict,
        'verdict_color': verdict_color,
        'revenue_growth_used': revenue_growth,
        'net_margin_used': net_margin,
        'cash_conversion_used': cash_conversion,
        'discount_rate_used': discount_rate,
        'exit_multiple_used': exit_multiple,
        'forecast_years': forecast_years,
        'latest_revenue': latest_revenue,
        'net_cash': net_cash,
        'revenue_projections': revenue_projections,
        'net_income_projections': net_income_projections,
        'fcfe_projections': fcfe_projections,
        'pv_fcfe': pv_fcfe,
        'total_pv_fcfe': total_pv_fcfe,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'shares_outstanding': shares,
        'company_name': metrics.get('company_name', ticker),
        'data_source': metrics.get('data_source', 'unknown'),
        'error': None,
        'method': 'Advanced DCF (Alpha Spread style)'
    }