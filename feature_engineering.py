# feature_engineering.py

import pandas as pd
import numpy as np
import yfinance as yf

# --- Sector ETFs (for sector momentum) ---
SECTOR_ETFS = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLRE', 'XLU']

def get_yield_data(start_date, end_date):
    """
    Fetch 10-year Treasury yield (^TNX) as a simple indicator.
    Returns a DataFrame with column 'TNX'.
    """
    try:
        tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
        if tnx.empty:
            return pd.DataFrame(columns=['TNX'])
        # Ensure it's a Series
        if isinstance(tnx, pd.DataFrame):
            tnx = tnx.iloc[:, 0]
        return pd.DataFrame({'TNX': tnx})
    except Exception as e:
        print(f"Yield data error: {e}")
        return pd.DataFrame(columns=['TNX'])

def get_sector_etf_data(start_date, end_date):
    """
    Fetch daily closes for sector ETFs and SPY.
    Returns DataFrame with columns: SPY_close, {sector}_close, and daily returns,
    plus rolling correlation with SPY.
    """
    all_tickers = ['SPY'] + SECTOR_ETFS
    data = {}
    # First, get a common index from SPY (most reliable)
    spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False)
    if spy_df.empty:
        # If SPY fails, return empty DataFrame with expected columns
        cols = ['SPY_close'] + [f'{etf}_close' for etf in SECTOR_ETFS] + \
               ['SPY_close_ret'] + [f'{etf}_close_ret' for etf in SECTOR_ETFS] + \
               [f'{etf}_corr_60d' for etf in SECTOR_ETFS]
        return pd.DataFrame(columns=cols)
    
    common_index = spy_df.index
    # Extract SPY close as a Series, ensure 1D
    spy_close = spy_df['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]  # take first column if it's a DataFrame
    data['SPY_close'] = spy_close

    for etf in SECTOR_ETFS:
        try:
            df = yf.download(etf, start=start_date, end=end_date, progress=False)
            if df.empty:
                series = pd.Series(index=common_index, dtype=float)
            else:
                # Extract close, ensure it's a Series
                close_series = df['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                # Reindex to common index, forward fill
                series = close_series.reindex(common_index, method='ffill')
        except Exception as e:
            series = pd.Series(index=common_index, dtype=float)
        data[f'{etf}_close'] = series

    # Build DataFrame from dictionary of Series
    df_sector = pd.DataFrame(data, index=common_index)

    # Compute returns
    returns = df_sector.pct_change().add_suffix('_ret')
    df_sector = pd.concat([df_sector, returns], axis=1)

    # Rolling correlation with SPY (60-day)
    spy_ret = df_sector['SPY_close_ret']
    for etf in SECTOR_ETFS:
        corr_col = f'{etf}_corr_60d'
        df_sector[corr_col] = df_sector[f'{etf}_close_ret'].rolling(60).corr(spy_ret)

    return df_sector

def get_vix_spot(start_date, end_date):
    """
    Fetch VIX spot price.
    Returns DataFrame with column 'VIX'.
    """
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
        if vix.empty:
            return pd.DataFrame(columns=['VIX'])
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        return pd.DataFrame({'VIX': vix})
    except:
        return pd.DataFrame(columns=['VIX'])

def get_macro_and_sector_data(start_date, end_date):
    """
    Master function that fetches all macro/sector data and returns a single DataFrame aligned by date.
    """
    vix_df = get_vix_spot(start_date, end_date)
    yield_df = get_yield_data(start_date, end_date)
    sector_df = get_sector_etf_data(start_date, end_date)

    # Start with vix_df index as base (if available), else use sector_df index
    if not vix_df.empty:
        base_index = vix_df.index
    elif not sector_df.empty:
        base_index = sector_df.index
    else:
        # No data at all – return empty DataFrame with expected columns
        all_cols = ['VIX', 'TNX'] + list(sector_df.columns) if not sector_df.empty else ['VIX', 'TNX']
        return pd.DataFrame(columns=all_cols)

    # Reindex all to base_index and forward fill
    vix_df = vix_df.reindex(base_index, method='ffill')
    yield_df = yield_df.reindex(base_index, method='ffill')
    sector_df = sector_df.reindex(base_index, method='ffill')

    combined = pd.concat([vix_df, yield_df, sector_df], axis=1)
    return combined

def add_enhanced_features(stock_df, ticker, macro_sector_df):
    """
    Takes a stock OHLCV DataFrame (with columns Open, High, Low, Close, Volume) and the macro_sector_df.
    Returns a copy with new features merged on the date index.
    """
    data = stock_df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.join(macro_sector_df, how='left')
    return data

def prepare_training_data(ticker_list, start_date, end_date):
    """
    For training: loop over tickers, download data, add ta features, add enhanced features,
    and stack into one big DataFrame with a 'ticker' column.
    """
    from ta import add_all_ta_features
    macro_sector = get_macro_and_sector_data(start_date, end_date)

    all_data = []
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
            df = add_enhanced_features(df, ticker, macro_sector)
            df['future_close'] = df['Close'].shift(-5)
            df['target'] = (df['future_close'] > df['Close']).astype(int)
            df['ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if all_data:
        combined = pd.concat(all_data, axis=0)
        combined = combined.dropna(subset=['target'])
        return combined
    else:
        return pd.DataFrame()