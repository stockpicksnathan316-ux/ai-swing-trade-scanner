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
import requests
import joblib
import feature_engineering as fe
import uuid
from datetime import datetime, timedelta
from supabase import create_client

# ---------- MUST BE THE FIRST STREAMLIT COMMAND ----------

st.set_page_config(page_title="AI Momentum Predictor", layout="wide")

# ---------- Helper function (safe TA) ----------
def safe_add_ta_features(df, min_rows=10):
    if df is None or len(df) < min_rows:
        st.warning(f"Insufficient data ({len(df) if df is not None else 0} rows) to compute technical indicators. Using raw price data only.")
        return df
    try:
        df_ta = df.copy()
        df_ta = add_all_ta_features(df_ta, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        return df_ta
    except Exception as e:
        st.warning(f"Technical indicator calculation failed: {str(e)}. Using raw price data only.")
        return df

# ---------- Stripe & Supabase Initialization ----------
stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]
price_id = st.secrets["stripe_price_id"]
base_url = st.secrets.get("base_url", "http://localhost:8501")

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

# ---------- User management functions (email‑based) ----------
def check_user_pro_status(email):
    """Check Pro status directly from Supabase users table."""
    if not email:
        return False
    try:
        response = supabase.table("users").select("is_pro").eq("email", email).execute()
        if response.data:
            return response.data[0]["is_pro"]
        else:
            # Create user record on first login
            supabase.table("users").insert({
                "email": email,
                "name": st.session_state.get("user_name", ""),
                "created_at": datetime.now().isoformat(),
                "is_pro": False
            }).execute()
            return False
    except Exception as e:
        print(f"Error checking Pro status: {e}")
        return False

def get_user_scans_used(email):
    """Get number of scans used by this user."""
    try:
        response = supabase.table("user_usage").select("scans_used").eq("email", email).execute()
        if response.data:
            return response.data[0]["scans_used"]
        else:
            supabase.table("user_usage").insert({
                "email": email,
                "scans_used": 0,
                "last_scan": datetime.now().isoformat()
            }).execute()
            return 0
    except Exception as e:
        print(f"Error getting scan count: {e}")
        return 0

def increment_user_scans(email):
    """Add 1 to the scan count for this user."""
    try:
        current = get_user_scans_used(email)
        supabase.table("user_usage").update({
            "scans_used": current + 1,
            "last_scan": datetime.now().isoformat()
        }).eq("email", email).execute()
    except Exception as e:
        print(f"Error incrementing scan count: {e}")

# ========== GOOGLE OAUTH AUTHENTICATION ==========
# Check if user is logged in
if not hasattr(st, 'experimental_user') or not st.experimental_user.get("is_logged_in", False):
    st.title("🔐 AI Stock Scanner")
    st.markdown("Please log in to continue:")
    if st.button("🚀 Login with Google"):
        st.login("google")
    st.stop()

# User is logged in
email = st.experimental_user.email
name = st.experimental_user.get("name", "User")
st.session_state["user_email"] = email
st.session_state["user_name"] = name

# Logout button in sidebar
with st.sidebar:
    st.markdown(f"👤 **{name}**")
    st.markdown(f"📧 {email}")
    if st.button("🚪 Logout"):
        st.logout()

# Check Pro status
st.session_state['paid_user'] = check_user_pro_status(email)

# ========== REST OF THE APP ==========
st.title("🤖 AI Momentum Predictor")

# Initialize session state for payment status (already done above, but ensure)
if 'paid_user' not in st.session_state:
    st.session_state.paid_user = False

# Handle Stripe return (from payment)
query_params = st.query_params.to_dict()
if "stripe_session_id" in query_params:
    session_id_raw = query_params["stripe_session_id"]
    if isinstance(session_id_raw, list):
        stripe_session_id = session_id_raw[0]
    else:
        stripe_session_id = session_id_raw
    try:
        session = stripe.checkout.Session.retrieve(stripe_session_id)
        if session.payment_status == "paid":
            st.session_state.paid_user = True
            st.success("🎉 Payment successful! You now have unlimited access.")
            st.query_params.clear()
        else:
            st.warning(f"Payment not completed. Status: {session.payment_status}")
    except Exception as e:
        st.error(f"❌ Error verifying payment: {e}")

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

# --- Show free‑tier status (using email) ---
if st.session_state.paid_user:
    st.sidebar.success("🌟 Premium subscriber - unlimited scans!")
else:
    scans_used = get_user_scans_used(email)
    remaining = max(0, 5 - scans_used)
    st.sidebar.info(f"📊 Free tier: {remaining}/5 scans remaining")

# --- Upgrade button (uses email from session) ---
if st.button("📈 Upgrade to Pro ($20/month)"):
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': st.secrets["stripe_price_id"],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=base_url + "?stripe_session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base_url + "?payment=cancelled",
            customer_email=email   # use logged‑in email
        )
        st.session_state.checkout_url = checkout_session.url
        st.markdown(f"👉 [Click here to complete payment]({checkout_session.url})")
    except Exception as e:
        st.error(f"❌ Error creating checkout session: {e}")

if "checkout_url" in st.session_state:
    url = st.session_state.checkout_url
    st.success("✅ Ready to subscribe! Click the button below to complete your payment.")
    st.link_button("💳 Pay $20/month and unlock unlimited scans", url)
    del st.session_state.checkout_url

# ------------------- MAIN CHART -------------------
ticker = st.text_input("Stock", "AAPL", key="main_ticker")
period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="main_period")

# Auto‑refresh
from streamlit_autorefresh import st_autorefresh
st.sidebar.checkbox("Auto-refresh every 5 min", key="auto_refresh")
if st.session_state.auto_refresh:
    st_autorefresh(interval=300000, key="auto_refresh_timer")

# --- Load data ---
df = yf.download(ticker, period=period, progress=False)
if df.empty:
    st.error(f"❌ No data found for {ticker} for period {period}. Please try another ticker or period.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df = safe_add_ta_features(df)

# --- Macro data ---
# Define macro symbols before use
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

macro_data = fetch_macro_data(period=period)
df = df.join(macro_data, how='left').ffill().bfill()

# --- Target and feature engineering ---
df['future_close'] = df['Close'].shift(-5)
df['target'] = (df['future_close'] > df['Close']).astype(int)
df_full = df.copy()
df_clean = df.dropna().copy()
feature_columns = [col for col in df_clean.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']]

X = df_clean[feature_columns]
y = df_clean['target']

# Train/test split
split_idx = int(len(df_clean) * 0.8)
df_train = df_clean.iloc[:split_idx].copy()
df_test = df_clean.iloc[split_idx:].copy()
X_train = df_train[feature_columns]
y_train = df_train['target']
X_test = df_test[feature_columns]
y_test = df_test['target']

# --- Ensemble training ---
xgb_param_grid = {'n_estimators': [50,100,200], 'max_depth': [2,3,4], 'learning_rate': [0.01,0.05,0.1]}
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
st.write(f"✅ Best XGBoost params: {xgb_grid.best_params_}")
st.write(f"✅ XGBoost CV accuracy: {xgb_grid.best_score_:.2%}")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

ensemble_model = VotingClassifier([('xgb', xgb_best), ('rf', rf_model), ('lgb', lgb_model)], voting='soft')
ensemble_model.fit(X_train, y_train)
st.write("🎯 **Ensemble model trained with XGBoost, Random Forest, and LightGBM**")

# --- Predictions and metrics ---
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class, zero_division=0)

latest_row = df_full[feature_columns].fillna(0).iloc[[-1]]
live_prob = ensemble_model.predict_proba(latest_row)[0][1]

# --- Candlestick chart ---
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
))
if 'rsi' in df.columns:
    buy_signals = df[df['rsi'] < 30]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close']*0.98,
            mode='markers', marker=dict(size=8, color='lime', symbol='triangle-up'),
            name='RSI Buy Signal'
        ))
st.plotly_chart(fig, width='stretch')

# --- Metrics display ---
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Test Accuracy", f"{acc:.1%}")
col2.metric("⚡ Test Precision", f"{prec:.1%}")
col3.metric("📊 Training Days", len(df_train))
col4, col5 = st.columns(2)
col4.metric("🔮 Today's 5-Day UP Probability", f"{live_prob:.1%}")
col5.metric("📅 Latest Data", str(df.index[-1].date()))

# --- Backtest expander ---
if len(X_test) > 0:
    y_test_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    y_test_pred_class = (y_pred_proba > 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc_test = accuracy_score(y_test, y_test_pred_class)
    prec_test = precision_score(y_test, y_test_pred_class, zero_division=0)
    rec_test = recall_score(y_test, y_test_pred_class, zero_division=0)
    f1_test = f1_score(y_test, y_test_pred_class, zero_division=0)

    df_test['pred_prob'] = y_test_pred_proba
    df_test['pred_class'] = y_test_pred_class
    df_test['actual'] = y_test.values

    with st.expander("📈 Backtest Performance (out‑of‑sample)"):
        st.metric("Test Accuracy", f"{acc_test:.1%}")
        st.metric("Test Precision", f"{prec_test:.1%}")
        st.metric("Test Recall", f"{rec_test:.1%}")
        st.metric("Test F1 Score", f"{f1_test:.1%}")
        df_test['return'] = df_test['Close'].pct_change().shift(-1)
        df_test['strategy_return'] = df_test['pred_class'] * df_test['return']
        df_test['cumulative_market'] = (1 + df_test['return']).cumprod()
        df_test['cumulative_strategy'] = (1 + df_test['strategy_return']).cumprod()

        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(x=df_test.index, y=df_test['cumulative_market'],
                                       mode='lines', name='Buy & Hold'))
        fig_back.add_trace(go.Scatter(x=df_test.index, y=df_test['cumulative_strategy'],
                                       mode='lines', name='AI Strategy'))
        fig_back.update_layout(title="Equity Curve (Test Period)", xaxis_title="Date", yaxis_title="Cumulative Return")
        st.plotly_chart(fig_back, use_container_width=True)

        trades = df_test[df_test['pred_class'] == 1]
        if len(trades) > 0:
            win_rate = (trades['strategy_return'] > 0).mean()
            st.metric("Win Rate (on trades)", f"{win_rate:.1%}")

# --- Feature importance ---
if st.checkbox("Show what XGBoost learned"):
    importance = pd.DataFrame({'feature': feature_columns, 'importance': xgb_best.feature_importances_})
    importance = importance.sort_values('importance', ascending=False).head(10)
    st.bar_chart(importance.set_index('feature'))

# ------------------- MULTI‑TICKER SCREENER -------------------
st.sidebar.header("🔍 Market Scanner (50+ Tickers)")

# Load tickers
tickers_df = pd.read_csv('tickers.csv')
TICKERS = dict(zip(tickers_df['Symbol'], tickers_df['Sector']))
sectors = ['All'] + sorted(set(TICKERS.values()))
selected_sector = st.sidebar.selectbox("Filter by sector", sectors)

if selected_sector == 'All':
    ticker_list = list(TICKERS.keys())
else:
    ticker_list = [t for t, s in TICKERS.items() if s == selected_sector]

st.sidebar.write(f"📊 Scanning **{len(ticker_list)}** tickers")

@st.cache_data(ttl=86400)
def get_macro_sector_data_cached(period="2y"):
    """Fetch macro and sector data for the given period using feature_engineering, plus CL, and ensure all required columns exist."""
    end = pd.Timestamp.now()
    if period == "6mo":
        start = end - pd.DateOffset(months=6)
    elif period == "1y":
        start = end - pd.DateOffset(years=1)
    elif period == "2y":
        start = end - pd.DateOffset(years=2)
    else:
        start = end - pd.DateOffset(years=1)

    # Get base macro/sector data (VIX, TNX, sector ETFs)
    macro_df = fe.get_macro_and_sector_data(start.date(), end.date())

    # Add CL (oil futures) – this was missing
    try:
        cl = yf.download('CL=F', start=start, end=end, progress=False)['Close']
        # Align to macro_df index
        cl = cl.reindex(macro_df.index, method='ffill')
        macro_df['CL'] = cl
    except Exception as e:
        # If CL fails, add a zero column with the same index
        if not macro_df.empty:
            macro_df['CL'] = 0
        else:
            # If macro_df is empty, create a dummy index and add zeros later
            macro_df = pd.DataFrame(index=pd.date_range(start=start, end=end, freq='B'))

    # --- Ensure all required sector columns are present (fill missing with zeros) ---
    # Required columns are those in feature_cols that start with 'XL' or 'SPY', plus VIX, TNX, CL
    required_cols = [col for col in feature_cols if col.startswith(('XL', 'SPY')) or col in ['VIX', 'TNX', 'CL']]
    for col in required_cols:
        if col not in macro_df.columns:
            macro_df[col] = 0  # fill missing with zeros

    return macro_df

@st.cache_data(ttl=21600)
def scan_tickers_fallback(tickers, macro_sector_df):
    results = []
    for ticker in tickers:
        try:
            df_t = yf.download(ticker, period="1y", progress=False)
            if df_t.empty:
                results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": "No data", "Prob": 0.0})
                continue
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.droplevel(1)
            df_t = safe_add_ta_features(df_t)
            df_t = fe.add_enhanced_features(df_t, ticker, macro_sector_df)
            feature_cols = [c for c in df_t.columns if c not in ['Open','High','Low','Close','Volume']]
            split = int(len(df_t) * 0.8)
            train = df_t.iloc[:split]
            X_train = train[feature_cols].fillna(0)
            y_train = (train['Close'].shift(-5) > train['Close']).astype(int).fillna(0)

            xgb_t = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
            rf_t = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
            lgb_t = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
            xgb_t.fit(X_train, y_train)
            rf_t.fit(X_train, y_train)
            lgb_t.fit(X_train, y_train)

            ensemble_t = VotingClassifier([('xgb', xgb_t), ('rf', rf_t), ('lgb', lgb_t)], voting='soft')
            ensemble_t.fit(X_train, y_train)

            latest = df_t[feature_cols].fillna(0).iloc[[-1]]
            prob = ensemble_t.predict_proba(latest)[0][1]
            results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": f"{prob:.1%}", "Prob": prob})
        except Exception as e:
            results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": "Error", "Prob": 0.0})
    return results

scan_button = st.sidebar.button("Scan Selected Tickers")
if scan_button:
    if st.session_state.paid_user:
        # Pro user – no limit
        pass
    else:
        scans_used = get_user_scans_used(email)
        if scans_used >= 5:
            st.sidebar.error("🔒 You've used all 5 free scans. Please upgrade to Pro for unlimited access!")
            st.stop()
        else:
            increment_user_scans(email)

    with st.spinner(f"Scanning {len(ticker_list)} tickers... this may take a minute."):
        macro_sector_df = get_macro_sector_data_cached("1y")
        results = scan_tickers_fallback(ticker_list, macro_sector_df)
        st.session_state.scanner_results = results
        st.rerun()

# --- Display scanner results ---
if st.session_state.get('scanner_results'):
    results = st.session_state.scanner_results
    df_results = pd.DataFrame(results)
    df_results['Prob'] = pd.to_numeric(df_results['Prob'], errors='coerce')
    df_results = df_results.sort_values('Prob', ascending=False).drop(columns='Prob')

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
    st.dataframe(styled_df, width='stretch')

    if st.button("📥 Export Scanner Results to CSV"):
        df_results.to_csv("scanner_results.csv", index=False)
        st.success("✅ Saved as scanner_results.csv on your Desktop!")