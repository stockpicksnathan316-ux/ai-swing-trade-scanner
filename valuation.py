import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st

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
    except:
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


# ------------------- Advanced DCF (Alpha Spread style) -------------------
def get_financial_metrics(ticker):
    """
    Fetch key financial metrics using TTM (trailing twelve months) data,
    plus forward estimates from stock.info.
    Returns a dict with both historical and forward-looking metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        income_annual = stock.financials
        income_quarterly = stock.quarterly_income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow

        # ----- TTM Revenue (quarterly, annual, info fallback) -----
        latest_revenue = None
        if income_quarterly is not None and not income_quarterly.empty:
            try:
                rev_q = income_quarterly.loc['Total Revenue'].iloc[:4]
                if len(rev_q) == 4:
                    latest_revenue = rev_q.sum()
            except:
                pass
        if latest_revenue is None or latest_revenue <= 0:
            try:
                latest_revenue = income_annual.loc['Total Revenue'].iloc[0]
            except:
                pass
        if latest_revenue is None or latest_revenue <= 0:
            latest_revenue = info.get('totalRevenue') or info.get('revenue') or 0

        # ----- TTM Net Income -----
        latest_net_income = None
        if income_quarterly is not None and not income_quarterly.empty:
            try:
                ni_q = income_quarterly.loc['Net Income'].iloc[:4]
                if len(ni_q) == 4:
                    latest_net_income = ni_q.sum()
            except:
                pass
        if latest_net_income is None or latest_net_income <= 0:
            try:
                latest_net_income = income_annual.loc['Net Income'].iloc[0]
            except:
                pass
        if latest_net_income is None or latest_net_income <= 0:
            latest_net_income = info.get('netIncomeToCommon') or 0

        # ----- Shares -----
        shares = info.get('sharesOutstanding', 1)
        if shares is None or shares <= 0:
            shares = info.get('floatShares') or 1

        # ----- Net Cash -----
        try:
            cash = balance.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance.index else 0
            short_term_debt = balance.loc['Short Term Debt'].iloc[0] if 'Short Term Debt' in balance.index else 0
            long_term_debt = balance.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance.index else 0
            total_debt = short_term_debt + long_term_debt
            net_cash = cash - total_debt
        except:
            net_cash = 0

        # ----- Current Price -----
        current_price = info.get('regularMarketPrice', 0)

        # ----- Historical growth rates (5-year CAGR) -----
        revenue_series = income_annual.loc['Total Revenue'] if 'Total Revenue' in income_annual.index else None
        net_income_series = income_annual.loc['Net Income'] if 'Net Income' in income_annual.index else None
        fcf_series = cashflow.loc['Free Cash Flow'] if 'Free Cash Flow' in cashflow.index else None

        def cagr(series, years=5):
            if series is None or len(series) < years:
                return None
            ser = series.iloc[:years].dropna()
            if len(ser) < years:
                return None
            first = ser.iloc[-1]
            last = ser.iloc[0]
            if first <= 0 or last <= 0:
                return None
            return (last / first) ** (1 / (len(ser)-1)) - 1

        rev_growth_hist = cagr(revenue_series)
        net_margin_hist = (net_income_series.iloc[0] / revenue_series.iloc[0]) if revenue_series is not None and net_income_series is not None else None
        cash_conversion_hist = (fcf_series.iloc[0] / net_income_series.iloc[0]) if fcf_series is not None and net_income_series is not None else None

        # ----- Forward estimates from info -----
        forward_rev_growth = info.get('revenueGrowth')  # e.g., 0.096 for AAPL
        forward_net_margin = info.get('profitMargins')   # trailing margin, used as proxy
        forward_earnings_growth = info.get('earningsGrowth')
        current_ps_ratio = info.get('priceToSalesTrailing12Months')

        return {
            'latest_revenue': latest_revenue,
            'latest_net_income': latest_net_income,
            'latest_fcf': fcf_series.iloc[0] if fcf_series is not None else None,
            'shares_outstanding': shares,
            'current_price': current_price,
            'net_cash': net_cash,
            'revenue_growth_hist': rev_growth_hist,
            'net_margin_hist': net_margin_hist,
            'cash_conversion_hist': cash_conversion_hist,
            'company_name': info.get('longName', ticker),
            # Forward-looking fields
            'forward_revenue_growth': forward_rev_growth,
            'forward_net_margin': forward_net_margin,
            'forward_earnings_growth': forward_earnings_growth,
            'current_ps_ratio': current_ps_ratio,
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
    Advanced DCF model following Alpha Spread's methodology:
    - Project Revenue using a CAGR
    - Apply Net Margin to get Net Income
    - Convert Net Income to FCFE using Cash Flow Conversion
    - Discount FCFE at the discount rate
    - Terminal value using an exit multiple on terminal year revenue
    - Add net cash to equity value
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

    # Project revenue for explicit forecast period
    revenue_projections = [latest_revenue * (1 + revenue_growth) ** (i+1) for i in range(forecast_years)]
    net_income_projections = [rev * net_margin for rev in revenue_projections]
    fcfe_projections = [ni * cash_conversion for ni in net_income_projections]

    # Discount FCFE to present value
    pv_fcfe = []
    for i, fcfe in enumerate(fcfe_projections):
        pv = fcfe / (1 + discount_rate) ** (i+1)
        pv_fcfe.append(pv)

    total_pv_fcfe = sum(pv_fcfe)

    # Terminal value using exit multiple on terminal year revenue
    terminal_revenue = revenue_projections[-1]
    terminal_value = terminal_revenue * exit_multiple
    pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

    # Enterprise value
    enterprise_value = total_pv_fcfe + pv_terminal

    # Add net cash to get equity value
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

    # Build result dictionary
    result = {
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
        'error': None,
        'method': 'Advanced DCF (Alpha Spread style)'
    }

    return result