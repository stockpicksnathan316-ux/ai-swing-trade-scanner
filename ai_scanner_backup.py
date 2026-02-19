import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from ta import add_all_ta_features
from sklearn.metrics import accuracy_score, precision_score

st.set_page_config(page_title="AI Momentum Predictor", layout="wide")
st.title("🤖 AI Momentum Predictor")

# ------------------- SIDEBAR / MAIN CHART -------------------
ticker = st.text_input("Stock", "AAPL", key="main_ticker")
period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="main_period")

# --- Auto-refresh (optional) ---
from streamlit_autorefresh import st_autorefresh

st.sidebar.checkbox("Auto-refresh every 5 min", key="auto_refresh")
if st.session_state.auto_refresh:
    st_autorefresh(interval=300000, key="auto_refresh_timer")

# --- Load main data ---
df = yf.download(ticker, period=period, progress=False)

# --- Flatten MultiIndex if present ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# --- Add all technical indicators ---
df = add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)

# --- Create target: price up in 5 days? ---
df['future_close'] = df['Close'].shift(-5)
df['target'] = (df['future_close'] > df['Close']).astype(int)

# --- Keep a copy of the full data for live prediction (features only, NaNs allowed) ---
df_full = df.copy()

# --- Drop rows with NaN for training (removes the last 5 rows and any indicator NaNs) ---
df_clean = df.dropna().copy()

# --- Define feature columns (all except price/volume/target) ---
feature_columns = [col for col in df_clean.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']]

X = df_clean[feature_columns]
y = df_clean['target']

# --- Walk-forward train/test split (80% train, 20% test) ---
split_idx = int(len(df_clean) * 0.8)
df_train = df_clean.iloc[:split_idx].copy()
df_test = df_clean.iloc[split_idx:].copy()

X_train = df_train[feature_columns]
y_train = df_train['target']
X_test = df_test[feature_columns]
y_test = df_test['target']

# --- Train model ---
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# --- Predict on test set ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba > 0.5).astype(int)

# --- Compute accuracy / precision ---
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class, zero_division=0)

# --- Live prediction (today) using the FULL data (last row, even if target missing) ---
latest_row = df_full[feature_columns].fillna(0).iloc[[-1]]  # double brackets to keep DataFrame
live_prob = model.predict_proba(latest_row)[0][1]

# --- Plot candlestick with RSI buy signals (optional) ---
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price'
))

if 'rsi' in df.columns:
    buy_signals = df[df['rsi'] < 30]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'] * 0.98,
            mode='markers',
            marker=dict(size=8, color='lime', symbol='triangle-up'),
            name='RSI Buy Signal'
        ))

st.plotly_chart(fig, use_container_width='stretch')

# --- Display main metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Test Accuracy", f"{acc:.1%}")
col2.metric("⚡ Test Precision", f"{prec:.1%}")
col3.metric("📊 Training Days", len(df_train))

col4, col5 = st.columns(2)
col4.metric("🔮 Today's 5-Day UP Probability", f"{live_prob:.1%}")
col5.metric("📅 Latest Data", str(df.index[-1].date()))

# --- Feature importance (optional) ---
if st.checkbox("Show what the AI learned"):
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    st.bar_chart(importance.set_index('feature'))

# ------------------- MULTI-TICKER SCREENER -------------------
st.sidebar.header("🔍 Market Scanner")
scan_tickers = st.sidebar.text_area("Tickers (comma separated)", "AAPL,MSFT,TSLA,NVDA,SPY")
scan_button = st.sidebar.button("Scan")

if scan_button:
    tickers = [t.strip() for t in scan_tickers.split(",") if t.strip()]
    results = []

    for ticker in tickers:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Download data
                df_t = yf.download(ticker, period="1y", progress=False)
                if isinstance(df_t.columns, pd.MultiIndex):
                    df_t.columns = df_t.columns.droplevel(1)

                # Add technical indicators
                df_t = add_all_ta_features(
                    df_t, open="Open", high="High", low="Low", close="Close",
                    volume="Volume", fillna=True
                )

                # Feature columns (exclude price/volume)
                feature_cols = [c for c in df_t.columns if c not in 
                                ['Open', 'High', 'Low', 'Close', 'Volume']]

                # Train/validation split (80/20)
                split = int(len(df_t) * 0.8)
                train = df_t.iloc[:split]
                # We don't need validation here, just training for prediction

                X_train = train[feature_cols].fillna(0)
                # Create target: price up in 5 days
                y_train = (train['Close'].shift(-5) > train['Close']).astype(int).fillna(0)

                # Quick model
                model_t = xgb.XGBClassifier(n_estimators=50, max_depth=2,
                                            learning_rate=0.05, random_state=42)
                model_t.fit(X_train, y_train)

                # Predict on latest row
                latest = df_t[feature_cols].fillna(0).iloc[[-1]]
                prob = model_t.predict_proba(latest)[0][1]

                results.append({"Ticker": ticker, "Signal": f"{prob:.1%}"})
            except Exception as e:
                results.append({"Ticker": ticker, "Signal": "Error"})

    # --- Display results with color coding ---
    st.subheader("📊 Scanner Results")

    def color_signal(val):
        try:
            pct = float(val.strip('%')) / 100
            if pct > 0.65:
                return 'background-color: #2e7d32; color: white; font-weight: bold;'
            elif pct < 0.35:
                return 'background-color: #c62828; color: white; font-weight: bold;'
            else:
                return 'background-color: #f5f5f5; color: #1e1e1e;'
        except:
            return 'background-color: #f5f5f5; color: #1e1e1e;'

    styled_df = pd.DataFrame(results).style.applymap(color_signal, subset=['Signal'])
    st.dataframe(styled_df)

    # --- Export button ---
    if st.button("📥 Export Results to CSV"):
        pd.DataFrame(results).to_csv("scanner_results.csv", index=False)
        st.success("Saved as scanner_results.csv on your Desktop!")