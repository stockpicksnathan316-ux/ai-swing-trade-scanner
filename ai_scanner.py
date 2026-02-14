import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from ta import add_all_ta_features
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV

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

# --- Hyperparameter tuning with GridSearchCV ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1]
}

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
st.write(f"✅ Best parameters: {grid_search.best_params_}")
st.write(f"✅ Best CV accuracy: {grid_search.best_score_:.2%}")

# --- Predict on test set with best model ---
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba > 0.5).astype(int)

# --- Compute accuracy / precision ---
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class, zero_division=0)

# --- Live prediction (today) using the FULL data ---
latest_row = df_full[feature_columns].fillna(0).iloc[[-1]]
live_prob = best_model.predict_proba(latest_row)[0][1]

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

# --- Backtest on test set (out-of-sample) ---
if len(X_test) > 0:
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred_class = (y_test_pred_proba > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc_test = accuracy_score(y_test, y_test_pred_class)
    prec_test = precision_score(y_test, y_test_pred_class, zero_division=0)
    rec_test = recall_score(y_test, y_test_pred_class, zero_division=0)
    f1_test = f1_score(y_test, y_test_pred_class, zero_division=0)

    # Store test predictions for plotting
    df_test['pred_prob'] = y_test_pred_proba
    df_test['pred_class'] = y_test_pred_class
    df_test['actual'] = y_test.values

    # --- Backtest expander ---
    with st.expander("📈 Backtest Performance (out‑of‑sample)"):
        st.metric("Test Accuracy", f"{acc_test:.1%}")
        st.metric("Test Precision", f"{prec_test:.1%}")
        st.metric("Test Recall", f"{rec_test:.1%}")
        st.metric("Test F1 Score", f"{f1_test:.1%}")

        # Simple equity curve: simulate trades based on predictions
        df_test['return'] = df_test['Close'].pct_change().shift(-1)  # next day return
        df_test['strategy_return'] = df_test['pred_class'] * df_test['return']
        df_test['cumulative_market'] = (1 + df_test['return']).cumprod()
        df_test['cumulative_strategy'] = (1 + df_test['strategy_return']).cumprod()

        # Plot equity curves
        import plotly.graph_objects as go
        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(x=df_test.index, y=df_test['cumulative_market'],
                                       mode='lines', name='Buy & Hold'))
        fig_back.add_trace(go.Scatter(x=df_test.index, y=df_test['cumulative_strategy'],
                                       mode='lines', name='AI Strategy'))
        fig_back.update_layout(title="Equity Curve (Test Period)", xaxis_title="Date", yaxis_title="Cumulative Return")
        st.plotly_chart(fig_back, use_container_width=True)

        # Win rate on trades
        trades = df_test[df_test['pred_class'] == 1]
        if len(trades) > 0:
            win_rate = (trades['strategy_return'] > 0).mean()
            st.metric("Win Rate (on trades)", f"{win_rate:.1%}")
else:
    with st.expander("📈 Backtest Performance (out‑of‑sample)"):
        st.info("Not enough test data to display backtest.")

# --- Feature importance (optional) ---
if st.checkbox("Show what the AI learned"):
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    st.bar_chart(importance.set_index('feature'))

# ------------------- MULTI-TICKER SCREENER -------------------
st.sidebar.header("🔍 Market Scanner")
scan_tickers = st.sidebar.text_area("Tickers (comma separated)", "AAPL,MSFT,TSLA,NVDA,SPY")
scan_button = st.sidebar.button("Scan")

# Initialize session state for results if not present
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = None

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

                X_train = train[feature_cols].fillna(0)
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

    # Store results in session state
    st.session_state.scanner_results = results

# --- Display results if they exist in session state ---
if st.session_state.scanner_results is not None:
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

    styled_df = pd.DataFrame(st.session_state.scanner_results).style.applymap(color_signal, subset=['Signal'])
    st.dataframe(styled_df)

    # --- Export button ---
    if st.button("📥 Export Results to CSV"):
        pd.DataFrame(st.session_state.scanner_results).to_csv("scanner_results.csv", index=False)
        st.success("✅ Saved as scanner_results.csv on your Desktop!")