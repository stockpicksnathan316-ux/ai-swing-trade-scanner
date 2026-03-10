# train_new_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import feature_engineering as fe
import yfinance as yf
from ta import add_all_ta_features
import os

# --- Configuration ---
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TEST_SIZE = 0.2  # last 20% for testing

# Load tickers
if os.path.exists('tickers.csv'):
    tickers_df = pd.read_csv('tickers.csv')
    TICKERS = tickers_df['Symbol'].tolist()
    print(f"Loaded {len(TICKERS)} tickers from tickers.csv")
else:
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'XOM']
    print("Using default tickers")

# --- Fetch macro/sector data once for the whole period ---
print("Fetching macro/sector data...")
macro_sector = fe.get_macro_and_sector_data(START_DATE, END_DATE)

# --- Prepare training data ---
all_data = []
for ticker in TICKERS:
    try:
        print(f"Processing {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Add TA features
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        # Add enhanced features (macro/sector)
        df = fe.add_enhanced_features(df, ticker, macro_sector)

        # Forward fill to handle any gaps in macro data, then fill remaining NaNs with 0
        df = df.ffill().fillna(0)

        # Create target (5-day forward return)
        df['future_close'] = df['Close'].shift(-5)
        df['target'] = (df['future_close'] > df['Close']).astype(int)

        # Drop rows where target is NaN (last 5 rows of each ticker)
        df = df.dropna(subset=['target'])

        # Add ticker column for reference
        df['ticker'] = ticker

        all_data.append(df)
    except Exception as e:
        print(f"Error with {ticker}: {e}")

if not all_data:
    raise ValueError("No data collected!")

# Combine all tickers
df_all = pd.concat(all_data, axis=0)
print(f"Total rows: {len(df_all)}")

# Define feature columns (exclude non-features)
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target', 'ticker']
feature_cols = [c for c in df_all.columns if c not in exclude_cols]
print(f"Number of features: {len(feature_cols)}")

X = df_all[feature_cols]
y = df_all['target']

# --- Train/test split (preserve time order) ---
# Sort by date first
df_all = df_all.sort_index()
X = df_all[feature_cols]
y = df_all['target']

split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- Train models ---

# 1. XGBoost with grid search (small grid for speed)
print("Training XGBoost...")
xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.05]
}
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
print(f"Best XGB params: {xgb_grid.best_params_}")
print(f"XGB CV accuracy: {xgb_grid.best_score_:.2%}")

# 2. Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 3. LightGBM
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

# 4. Ensemble
print("Training Ensemble...")
ensemble = VotingClassifier(
    estimators=[('xgb', xgb_best), ('rf', rf_model), ('lgb', lgb_model)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# --- Evaluate ---
y_pred = ensemble.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy:.2%}")

# --- Save model and feature list ---
joblib.dump(ensemble, 'ensemble_model_v3.pkl')
joblib.dump(feature_cols, 'feature_cols_v3.pkl')
print("Model saved as ensemble_model_v3.pkl")