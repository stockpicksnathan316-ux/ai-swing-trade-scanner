# train_new_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import feature_engineering as fe
import os

# --- Configuration ---
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TEST_SIZE = 0.2  # 20% of data for testing (last 20% in time order)

# Load tickers from CSV if exists, else use a small list for testing
if os.path.exists('tickers.csv'):
    tickers_df = pd.read_csv('tickers.csv')
    TICKERS = tickers_df['Symbol'].tolist()
    print(f"Loaded {len(TICKERS)} tickers from tickers.csv")
else:
    # Fallback to a few tickers
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'XOM']
    print(f"tickers.csv not found, using default list: {TICKERS}")

# --- Prepare data ---
print("Preparing training data... (this may take a while)")
df = fe.prepare_training_data(TICKERS, START_DATE, END_DATE)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# --- Define features ---
# Exclude columns that are not features
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target', 'ticker']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Number of features: {len(feature_cols)}")

# Drop rows with NaN in features (may happen due to indicators at start of series)
df_clean = df.dropna(subset=feature_cols).copy()
print(f"Rows after dropping NaN: {len(df_clean)}")

# Sort by date (important for time-based split)
df_clean = df_clean.sort_index()

X = df_clean[feature_cols]
y = df_clean['target']

# --- Train/test split (preserve time order) ---
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- Train models ---

# 1. XGBoost with GridSearch (similar to original but maybe smaller grid for speed)
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

# 4. Ensemble (Voting Classifier)
print("Training Ensemble...")
ensemble = VotingClassifier(
    estimators=[('xgb', xgb_best), ('rf', rf_model), ('lgb', lgb_model)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# --- Evaluate on test set ---
y_pred = ensemble.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy:.2%}")

# --- Save model and feature list ---
joblib.dump(ensemble, 'ensemble_model_v2.pkl')
joblib.dump(feature_cols, 'feature_cols_v2.pkl')
print("Model and feature list saved.")