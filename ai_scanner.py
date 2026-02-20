import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from ta import add_all_ta_features
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import stripe
import os

# Initialize Stripe
stripe.api_key = st.secrets["stripe_secret_key"]
price_id = st.secrets["stripe_price_id"]
base_url = st.secrets.get("base_url", "http://localhost:8501")  # fallback for local dev

# --- Master ticker list with sectors ---
TICKERS = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'META': 'Technology',
    'NVDA': 'Technology',
    'AMD': 'Technology',
    'INTC': 'Technology',
    'TSLA': 'Consumer Cyclical',
    'AMZN': 'Consumer Cyclical',
    'NFLX': 'Communication Services',
    'DIS': 'Communication Services',
    'JPM': 'Financial',
    'BAC': 'Financial',
    'WFC': 'Financial',
    'GS': 'Financial',
    'V': 'Financial',
    'MA': 'Financial',
    'JNJ': 'Healthcare',
    'PFE': 'Healthcare',
    'MRK': 'Healthcare',
    'ABT': 'Healthcare',
    'UNH': 'Healthcare',
    'XOM': 'Energy',
    'CVX': 'Energy',
    'COP': 'Energy',
    'SLB': 'Energy',
    'BA': 'Industrials',
    'CAT': 'Industrials',
    'GE': 'Industrials',
    'HON': 'Industrials',
    'WMT': 'Consumer Defensive',
    'PG': 'Consumer Defensive',
    'KO': 'Consumer Defensive',
    'PEP': 'Consumer Defensive',
    'NEE': 'Utilities',
    'DUK': 'Utilities',
    'SO': 'Utilities',
    'PLD': 'Real Estate',
    'AMT': 'Real Estate',
    'EQIX': 'Real Estate',
}
# --- Macro data symbols ---
MACRO_SYMBOLS = {
    'VIX': '^VIX',
    'TNX': '^TNX',
    'CL': 'CL=F',
}

@st.cache_data(ttl=21600)
def fetch_macro_data(period="1y"):
    macro_df = pd.DataFrame()
    for name, symbol in MACRO_SYMBOLS.items():
        try:
            data = yf.download(symbol, period=period, progress=False)
            if not data.empty:
                macro_df[name] = data['Close']
        except:
            macro_df[name] = pd.Series(dtype='float64')
    macro_df = macro_df.ffill().bfill()
    return macro_df

st.set_page_config(page_title="AI Momentum Predictor", layout="wide")
st.title("🤖 AI Momentum Predictor")

# Initialize session state for free tier and payment status
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0
if 'paid_user' not in st.session_state:
    st.session_state.paid_user = False

# Handle Stripe return (this should be before any other UI)
query_params = st.query_params.to_dict()

if "session_id" in query_params:
    session_id_raw = query_params["session_id"]
    if isinstance(session_id_raw, list):
        session_id = session_id_raw[0]
    else:
        session_id = session_id_raw
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            st.session_state.paid_user = True
            st.success("🎉 Payment successful! You now have unlimited access.")
            st.query_params.clear()
        else:
            st.warning("Payment not completed. Please try again.")
    except Exception as e:
        st.error("❌ Error verifying payment. Please contact support.")

if "payment" in query_params:
    payment_raw = query_params["payment"]
    if isinstance(payment_raw, list):
        payment_val = payment_raw[0]
    else:
        payment_val = payment_raw
    if payment_val == "cancelled":
        st.info("Payment cancelled. You can still use the free tier.")
        st.query_params.clear()

# --- License key input (sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Unlock Unlimited Scans")
license_key_input = st.sidebar.text_input("Enter license key", type="password")
if st.sidebar.button("Activate License"):
    valid_keys = st.secrets.get("license_keys", ["test123"])
    if license_key_input in valid_keys:
        st.session_state.paid_user = True
        st.sidebar.success("License activated! You now have unlimited scans.")
        st.rerun()
    else:
        st.sidebar.error("Invalid license key")

# Show status (sidebar)
if st.session_state.paid_user:
    st.sidebar.success("Premium subscriber - unlimited scans!")
else:
    remaining = max(0, 5 - st.session_state.scan_count)
    st.sidebar.info(f"Free tier: {remaining}/5 scans remaining")

# If not paid and scans exhausted, show error and upgrade button
if st.session_state.scan_count >= 5 and not st.session_state.get("paid_user", False):
    st.error("⚠️ You've used all 5 free scans. Subscribe for unlimited access!")
    
    if st.button("📈 Upgrade to Pro ($20/month)"):
        try:
            checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
            'price': price_id,
            'quantity': 1,
            }],
            mode='subscription',
            success_url= base_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url= base_url + "?payment=cancelled",
            )
            # Show clickable link
            st.markdown(f"👉 [Click here to complete payment]({checkout_session.url})")            
        except Exception as e:
            st.error(f"❌ Error: {e}")

# After button click, show a prominent button to go to Stripe
if "checkout_url" in st.session_state:
    url = st.session_state.checkout_url
    st.success("✅ Ready to subscribe! Click the button below to complete your payment.")
    st.link_button("💳 Pay $20/month and unlock unlimited scans", url)
    # Optionally, you can clear the stored URL after some time, but leaving it for the session is fine.
    
# If we have a stored checkout URL, redirect to it
if "checkout_url" in st.session_state:
    url = st.session_state.checkout_url
    del st.session_state.checkout_url  # clear it so we don't redirect again
    st.components.v1.html(f'<script>window.location.replace("{url}");</script>', height=0, width=0)        

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

# --- Merge macro data ---
macro_data = fetch_macro_data(period=period)
df = df.join(macro_data, how='left').ffill().bfill()

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

# --- ENSEMBLE: XGBoost + Random Forest + LightGBM ---

# 1. Tuned XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid,
                        cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
st.write(f"✅ Best XGBoost params: {xgb_grid.best_params_}")
st.write(f"✅ XGBoost CV accuracy: {xgb_grid.best_score_:.2%}")

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 3. LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

# --- Voting Classifier (soft voting = average probabilities) ---
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_best),
        ('rf', rf_model),
        ('lgb', lgb_model)
    ],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

st.write("🎯 **Ensemble model trained with XGBoost, Random Forest, and LightGBM**")

# --- Predict on test set with ensemble ---
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba > 0.5).astype(int)

# --- Compute accuracy / precision ---
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class, zero_division=0)

# --- Live prediction (today) using the FULL data ---
latest_row = df_full[feature_columns].fillna(0).iloc[[-1]]
live_prob = ensemble_model.predict_proba(latest_row)[0][1]

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
    y_test_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
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
if st.checkbox("Show what XGBoost learned"):
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': xgb_best.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    st.bar_chart(importance.set_index('feature'))

# ------------------- MULTI‑TICKER SCREENER (UPGRADED) -------------------
st.sidebar.header("🔍 Market Scanner (50+ Tickers)")

# Sector filter
sectors = ['All'] + sorted(set(TICKERS.values()))
selected_sector = st.sidebar.selectbox("Filter by sector", sectors)

# Button to start scan
scan_button = st.sidebar.button("Scan Selected Tickers")

# Create a list of tickers based on sector filter
if selected_sector == 'All':
    ticker_list = list(TICKERS.keys())
else:
    ticker_list = [t for t, s in TICKERS.items() if s == selected_sector]

# Show count
st.sidebar.write(f"📊 Scanning **{len(ticker_list)}** tickers")

# --- Cached scanner function (refreshes every 6 hours) ---
@st.cache_data(ttl=21600)  # 6 hours in seconds
def scan_tickers(tickers, macro_df):
    results = []
    for ticker in tickers:
        try:
            df_t = yf.download(ticker, period="1y", progress=False)
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.droplevel(1)

            df_t = add_all_ta_features(
                df_t, open="Open", high="High", low="Low", close="Close",
                volume="Volume", fillna=True
            )

            # --- Merge macro data ---
            df_t = df_t.join(macro_df, how='left').ffill().bfill()

            feature_cols = [c for c in df_t.columns if c not in 
                            ['Open','High','Low','Close','Volume']]

            split = int(len(df_t) * 0.8)
            train = df_t.iloc[:split]

            X_train = train[feature_cols].fillna(0)
            y_train = (train['Close'].shift(-5) > train['Close']).astype(int).fillna(0)

            # Quick ensemble for scanner (use same structure as main model)
            xgb_t = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
            rf_t = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
            lgb_t = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)

            xgb_t.fit(X_train, y_train)
            rf_t.fit(X_train, y_train)
            lgb_t.fit(X_train, y_train)

            ensemble_t = VotingClassifier(
                estimators=[('xgb', xgb_t), ('rf', rf_t), ('lgb', lgb_t)],
                voting='soft'
            )
            ensemble_t.fit(X_train, y_train)

            latest = df_t[feature_cols].fillna(0).iloc[[-1]]
            prob = ensemble_t.predict_proba(latest)[0][1]

            results.append({
                "Ticker": ticker,
                "Sector": TICKERS[ticker],
                "Signal": f"{prob:.1%}",
                "Prob": prob
            })
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Sector": TICKERS.get(ticker, "Unknown"),
                "Signal": "Error",
                "Prob": 0.0
            })
    return results

# --- When scan button is clicked ---
if scan_button:
    with st.spinner(f"Scanning {len(ticker_list)} tickers... this may take a minute."):
        macro_data = fetch_macro_data(period="1y")
        results = scan_tickers(ticker_list, macro_data)
        st.session_state.scan_count += 1

    # Store in session state so it persists
    st.session_state.scanner_results = results

# --- Display results if they exist ---
if st.session_state.get('scanner_results'):
    results = st.session_state.scanner_results

    # Create DataFrame and sort by probability
    df_results = pd.DataFrame(results)
    df_results['Prob'] = pd.to_numeric(df_results['Prob'], errors='coerce')
    df_results = df_results.sort_values('Prob', ascending=False).drop(columns='Prob')

    # --- TOP BULLISH / BEARISH BOXES ---
    col_bull, col_bear = st.columns(2)
    bullish = df_results.iloc[0] if len(df_results) > 0 else None
    bearish = df_results.iloc[-1] if len(df_results) > 1 else None

    with col_bull:
        st.success("🔥 **Most Bullish**")
        if bullish is not None:
            st.metric(bullish['Ticker'], bullish['Signal'], delta=None)
            st.caption(f"Sector: {bullish['Sector']}")
        else:
            st.write("No data")

    with col_bear:
        st.error("🥶 **Most Bearish**")
        if bearish is not None:
            st.metric(bearish['Ticker'], bearish['Signal'], delta=None)
            st.caption(f"Sector: {bearish['Sector']}")
        else:
            st.write("No data")

    # --- COLOR‑CODED TABLE ---
    st.subheader("📊 Full Scanner Results")

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

    styled_df = df_results.style.applymap(color_signal, subset=['Signal'])
    st.dataframe(styled_df, use_container_width=True)

    # --- EXPORT BUTTON ---
    if st.button("📥 Export Scanner Results to CSV"):
        df_results.to_csv("scanner_results.csv", index=False)
        st.success("✅ Saved as scanner_results.csv on your Desktop!")