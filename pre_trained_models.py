import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import hashlib
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from ta import add_all_ta_features
import feature_engineering as fe
import logging
from datetime import datetime

log_filename = f"pre_train_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

                    
# ------------------- Helper: safe technical indicators -------------------
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

# ------------------- Caching helpers -------------------
def get_stock_model_cache_path(ticker):
    ticker_hash = hashlib.md5(ticker.encode()).hexdigest()
    cache_dir = "tick_sniper_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    model_path = os.path.join(cache_dir, f"stock_model_{ticker_hash}.pkl")
    features_path = os.path.join(cache_dir, f"stock_features_{ticker_hash}.pkl")
    return model_path, features_path

def is_model_fresh(ticker, max_age_hours=24):
    model_path, _ = get_stock_model_cache_path(ticker)
    if not os.path.exists(model_path):
        return False
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    age = datetime.now() - mod_time
    return age < timedelta(hours=max_age_hours)

def get_stock_specific_model(ticker, df_basic, force_retrain=False):
    """
    Trains and caches the full ensemble model (XGB+RF+LGB) for a ticker.
    This is the same function used in the main app.
    """
    model_path, features_path = get_stock_model_cache_path(ticker)

    if not force_retrain and is_model_fresh(ticker):
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
        return model, feature_cols

    # --- Train the model (same logic as alpha=0 branch) ---
    df = df_basic.copy()
    df['future_close'] = df['Close'].shift(-5)
    df['target'] = (df['future_close'] > df['Close']).astype(int)
    df_clean = df.dropna(subset=['target']).copy()

    feature_columns = [col for col in df_clean.columns if col not in
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']]

    X = df_clean[feature_columns]
    y = df_clean['target']

    split_idx = int(len(df_clean) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    # XGBoost with grid search
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'scale_pos_weight': [1, 1.5, 2, 3]
    }
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid,
                            cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)

    ensemble_model = VotingClassifier(
        estimators=[('xgb', xgb_best), ('rf', rf_model), ('lgb', lgb_model)],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)

    # Save model and feature columns
    joblib.dump(ensemble_model, model_path)
    joblib.dump(feature_columns, features_path)
    return ensemble_model, feature_columns

# ------------------- Macro data for basic features -------------------
def get_basic_macro_data(period="1y"):
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
    # Return only the basic macro columns used for stock‑specific models
    return macro_df[['VIX', 'TNX', 'CL']]

# ------------------- Main -------------------
if __name__ == "__main__":
    # Load ticker list
    tickers_df = pd.read_csv('tickers.csv')
    tickers = tickers_df['Symbol'].tolist()
    print(f"Pre‑training models for {len(tickers)} tickers...")

    # Get basic macro data (VIX, TNX, CL)
    basic_macro_df = get_basic_macro_data("1y")

    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            # Download 1 year of data
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty:
                print(f"  No data for {ticker}, skipping.")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Add technical indicators
            df = safe_add_ta_features(df)

            # Add basic macro columns
            df_basic = df.join(basic_macro_df, how='left').ffill().bfill()

            # Train and cache the stock‑specific model
            get_stock_specific_model(ticker, df_basic, force_retrain=True)
            print(f"  Done.")
        except Exception as e:
            print(f"  Error: {e}")

    print("Pre‑training complete.")