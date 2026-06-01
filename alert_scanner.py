import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from datetime import datetime, timedelta
from supabase import create_client
import smtplib
from email.mime.text import MIMEText
import feature_engineering as fe
from ta import add_all_ta_features
from dotenv import load_dotenv
load_dotenv()

# --- Environment variables (set these in your system or use a .env file) ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
EMAIL_SENDER = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp-relay.brevo.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load models and feature columns ---
pooled_model = joblib.load("pooled_model.pkl")
pooled_feature_cols = joblib.load("pooled_feature_cols.pkl")
calibration_map = joblib.load("calibration_map.pkl")   # optional
print("Models loaded.")

# --- Helper functions (copied from main app) ---
def safe_add_ta_features(df, min_rows=10):
    if df is None or len(df) < min_rows:
        return df
    try:
        df_ta = df.copy()
        df_ta = add_all_ta_features(df_ta, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        return df_ta
    except Exception as e:
        print(f"TA failed: {e}")
        return df

def calibrate_prob(prob, cal_map):
    if cal_map is None:
        return prob
    bins = cal_map['bin_edges']
    win_rates = cal_map['win_rates']
    for i in range(len(bins)-1):
        if bins[i] <= prob < bins[i+1]:
            return win_rates[i]
    return prob

def get_macro_sector_data(period="1y"):
    end = pd.Timestamp.now()
    if period == "1y":
        start = end - pd.DateOffset(years=1)
    elif period == "2y":
        start = end - pd.DateOffset(years=2)
    else:
        start = end - pd.DateOffset(months=6)
    macro_df = fe.get_macro_and_sector_data(start.date(), end.date())
    try:
        cl = yf.download('CL=F', start=start, end=end, progress=False)['Close']
        cl = cl.reindex(macro_df.index, method='ffill')
        macro_df['CL'] = cl
    except:
        macro_df['CL'] = 0
    # Ensure required columns exist (fill with 0)
    required_cols = [col for col in pooled_feature_cols if col.startswith(('XL', 'SPY')) or col in ['VIX', 'TNX', 'CL']]
    for col in required_cols:
        if col not in macro_df.columns:
            macro_df[col] = 0
    return macro_df

def get_pooled_prediction(ticker, macro_df, sector_etf):
    print(f"  get_pooled_prediction for {ticker}")
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        print("  No data downloaded")
        return None
    print(f"  Data downloaded, shape: {df.shape}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = safe_add_ta_features(df)
    fundamentals = fe.get_fundamentals(ticker)
    df = fe.add_enhanced_features(df, ticker, macro_df, sector_etf, fundamentals)
    # Align features
    for col in pooled_feature_cols:
        if col not in df.columns:
            df[col] = 0
    latest = df[pooled_feature_cols].fillna(0).iloc[[-1]]
    prob_raw = pooled_model.predict_proba(latest)[0][1]
    print(f"  Raw pooled probability: {prob_raw:.3f}")
    return prob_raw   # return raw, not calibrated

def get_hybrid_prediction(ticker, macro_df, sector_etf, alpha):
    print(f"  Entering get_hybrid_prediction for {ticker}")
    # First, get pooled prediction
    pooled_prob = get_pooled_prediction(ticker, macro_df, sector_etf)
    print(f"  pooled_prob = {pooled_prob}")
    if pooled_prob is None:
        return None

    # Now stock‑specific model (train on the fly)
    print(f"  Downloading data for stock model")
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        print("  No data for stock model")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    print(f"  Stock data shape: {df.shape}")
    df = safe_add_ta_features(df)
    df = fe.add_enhanced_features(df, ticker, macro_df, sector_etf, {})   # no fundamentals needed for training
    df['future_close'] = df['Close'].shift(-5)
    df['target'] = (df['future_close'] > df['Close']).astype(int)
    df_clean = df.dropna(subset=['target']).copy()
    print(f"  After cleaning, rows: {len(df_clean)}")
    if len(df_clean) < 20:
        print("  Not enough rows for stock model")
        return None
    exclude_cols = ['Open','High','Low','Close','Volume','future_close','target','ticker']
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
    split_idx = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split_idx]
    X_train = train[feature_cols].fillna(0)
    y_train = train['target']
    print(f"  Training stock model on {X_train.shape[0]} rows, {X_train.shape[1]} features")
    stock_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    stock_model.fit(X_train, y_train)
    latest_stock = df_clean[feature_cols].fillna(0).iloc[[-1]]
    stock_prob = stock_model.predict_proba(latest_stock)[0][1]
    print(f"  stock_prob = {stock_prob:.3f}")
    blended = alpha * pooled_prob + (1 - alpha) * stock_prob
    print(f"  blended = {blended:.3f}")
    return blended   # FIX: return blended probability

def send_alert_email(user_email, ticker, prob, threshold, model_type):
    subject = f"AI Stock Alert: {ticker} ({model_type} model)"
    body = (
        f"Your watched stock {ticker} has a probability of {prob:.1%}\n"
        f"which meets your threshold of {threshold:.1%}.\n"
        f"Check the app for details."
    )
    msg = MIMEText(body, 'plain')
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = user_email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, [user_email], msg.as_string())

def main():
    # Fetch all alerts
    alerts = supabase.table("user_alerts").select("*").execute().data
    print(f"Found {len(alerts)} alerts.")
    if not alerts:
        print("No alerts found.")
        return

    # Load mapping from ticker to sector (from tickers.csv)
    tickers_df = pd.read_csv('tickers.csv')
    sector_map = dict(zip(tickers_df['Symbol'], tickers_df['Sector']))
    sector_to_etf = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Communication Services': 'XLC',
        'Industrials': 'XLI',
        'Consumer Defensive': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Basic Materials': 'XLB',
        'Broad Market ETFs': 'SPY'
    }

    macro_df = get_macro_sector_data()

    for alert in alerts:
        model_type = alert['model_type']
        print(f"Processing alert for {alert['ticker']} (model_type={model_type})")
        # Skip old scanner for now (would need pre‑trained models)
        if model_type == 'old':
            print(f"Skipping old scanner alert for {alert['ticker']} (not implemented yet).")
            continue

        ticker = alert['ticker']
        threshold = alert['threshold']
        alpha = alert['alpha']
        user_email = alert['user_email']

        # Cooldown: skip if last_triggered within last 24h
        last = alert.get('last_triggered')
        if last and (datetime.now() - datetime.fromisoformat(last)).days < 1:
            continue

        sector = sector_map.get(ticker, 'Unknown')
        sector_etf = sector_to_etf.get(sector, None)
        print(f"  Sector: {sector}, ETF: {sector_etf}")

        if model_type == 'pooled':
            prob = get_pooled_prediction(ticker, macro_df, sector_etf)
        elif model_type == 'hybrid':
            prob = get_hybrid_prediction(ticker, macro_df, sector_etf, alpha)
        else:
            continue

        if prob is not None and prob >= threshold:
            send_alert_email(user_email, ticker, prob, threshold, model_type)
            # Update last_triggered
            supabase.table("user_alerts").update({"last_triggered": datetime.now().isoformat()}).eq("id", alert['id']).execute()
            print(f"Alert sent for {ticker} to {user_email}")
        else:
            print(f"No alert for {ticker}: prob={prob}, threshold={threshold}")

if __name__ == "__main__":
    main()
    print("Script finished.")