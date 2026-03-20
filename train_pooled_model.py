# train_pooled_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import feature_engineering as fe
from ta import add_all_ta_features
import yfinance as yf

# --- Configuration ---
START_DATE = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')   # last 3 years
END_DATE = datetime.now().strftime('%Y-%m-%d')
TICKERS_CSV = 'tickers.csv'
MODEL_SAVE_PATH = 'pooled_model.pkl'
FEATURE_COLS_SAVE_PATH = 'pooled_feature_cols.pkl'

# --- Load tickers ---
tickers_df = pd.read_csv(TICKERS_CSV)
ticker_list = tickers_df['Symbol'].tolist()
print(f"Processing {len(ticker_list)} tickers...")

# --- Collect training data ---
print("Collecting training data...")
data = fe.prepare_training_data(ticker_list, START_DATE, END_DATE)
if data.empty:
    raise ValueError("No data collected. Exiting.")

print(f"Total rows: {len(data)}")
print(f"Features available: {[c for c in data.columns if c not in ['ticker', 'target', 'future_close', 'Close', 'Open', 'High', 'Low', 'Volume']][:5]}...")

# --- Define features and target ---
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target', 'ticker']
feature_cols = [c for c in data.columns if c not in exclude_cols]
X = data[feature_cols].fillna(0)
y = data['target']

print(f"Training on {X.shape[0]} rows with {len(feature_cols)} features.")

# --- Train ensemble ---
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

ensemble = VotingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model), ('lgb', lgb_model)],
    voting='soft'
)

print("Training ensemble...")
ensemble.fit(X, y)

# --- Save model and feature columns ---
joblib.dump(ensemble, MODEL_SAVE_PATH)
joblib.dump(feature_cols, FEATURE_COLS_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"Feature columns saved to {FEATURE_COLS_SAVE_PATH}")