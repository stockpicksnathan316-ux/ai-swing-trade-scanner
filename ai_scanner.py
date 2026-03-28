import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
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
from datetime import datetime, timedelta
from supabase import create_client
from feature_engineering import get_fundamentals
import hashlib

# ------------------- Helper: safe technical indicators -------------------
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

# --- Stock‑specific model caching ---
def get_stock_model_cache_path(ticker):
    """Return file paths for cached model and feature columns."""
    ticker_hash = hashlib.md5(ticker.encode()).hexdigest()
    model_path = f"stock_model_{ticker_hash}.pkl"
    features_path = f"stock_features_{ticker_hash}.pkl"
    return model_path, features_path

def is_model_fresh(ticker, max_age_hours=24):
    """Check if cached model exists and is younger than max_age_hours."""
    model_path, _ = get_stock_model_cache_path(ticker)
    if not os.path.exists(model_path):
        return False
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    age = datetime.now() - mod_time
    return age < timedelta(hours=max_age_hours)

def get_stock_specific_model(ticker, df_basic, force_retrain=False):
    """
    Returns a trained ensemble model (XGB+RF+LGB) for the given ticker,
    using the DataFrame that already contains technical indicators and basic macro.
    Caches the model with 24‑hour freshness.
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



# ------------------- Supabase client -------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

# Restore session if we have valid tokens in session state
if 'supabase_session' in st.session_state and st.session_state.supabase_session is not None:
    session = st.session_state.supabase_session
    # Ensure session has the required keys
    if 'access_token' in session and 'refresh_token' in session:
        supabase.auth.set_session(session['access_token'], session['refresh_token'])

# ------------------- Email‑based tracking functions -------------------
def get_user_scans_used(email):
    """Get number of scans used for this user from free_trial_usage table."""
    try:
        response = supabase.table("free_trial_usage") \
            .select("scans_used") \
            .eq("email", email) \
            .execute()
        if response.data:
            return response.data[0]["scans_used"]
        else:
            # First scan for this user – insert row with 0
            supabase.table("free_trial_usage").insert({
                "email": email,
                "scans_used": 0,
                "last_scan": datetime.now().isoformat()
            }).execute()
            return 0
    except Exception as e:
        st.error(f"Error getting scan count: {e}")
        return 0

def increment_user_scans(email):
    """Add 1 to the scan count for this user."""
    try:
        # Get current count
        current = get_user_scans_used(email)
        new_count = current + 1
        supabase.table("free_trial_usage") \
            .update({
                "scans_used": new_count,
                "last_scan": datetime.now().isoformat()
            }) \
            .eq("email", email) \
            .execute()
    except Exception as e:
        st.error(f"Error incrementing scan count: {e}")

def check_user_pro_status(email):
    """Check Pro status directly from paid_users table."""
    if not email:
        return False
    try:
        response = supabase.table("paid_users").select("is_pro").eq("email", email).execute()
        if response.data:
            return response.data[0]["is_pro"]
        else:
            # User not found – create a record
            supabase.table("paid_users").insert({
                "email": email,
                "name": st.session_state.get("user_name", ""),
                "created_at": datetime.now().isoformat(),
                "is_pro": False
            }).execute()
            return False
    except Exception as e:
        st.error(f"Error checking Pro status: {e}")
        return False

# ------------------- Load tickers -------------------
tickers_df = pd.read_csv('tickers.csv')
TICKERS = dict(zip(tickers_df['Symbol'], tickers_df['Sector']))

# Sector to ETF mapping
sector_to_etf = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY',
    'Communication Services': 'XLC',   # may need to add XLC to SECTOR_ETFS list
    'Industrials': 'XLI',
    'Consumer Defensive': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Basic Materials': 'XLB',
    'Broad Market ETFs': 'SPY'          # for ETFs, compare to market itself
}

# ------------------- Macro symbols -------------------
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

    macro_df = fe.get_macro_and_sector_data(start.date(), end.date())

    try:
        cl = yf.download('CL=F', start=start, end=end, progress=False)['Close']
        cl = cl.reindex(macro_df.index, method='ffill')
        macro_df['CL'] = cl
    except Exception as e:
        if not macro_df.empty:
            macro_df['CL'] = 0
        else:
            macro_df = pd.DataFrame(index=pd.date_range(start=start, end=end, freq='B'))

    # Ensure all required sector columns exist
    required_cols = [col for col in feature_cols if col.startswith(('XL', 'SPY')) or col in ['VIX', 'TNX', 'CL']]
    for col in required_cols:
        if col not in macro_df.columns:
            macro_df[col] = 0

    return macro_df

# ------------------- Load model -------------------
if os.path.exists('ensemble_model_v3.pkl'):
    ensemble_model = joblib.load('ensemble_model_v3.pkl')
    feature_cols = joblib.load('feature_cols_v3.pkl')
    xgb_model = ensemble_model.named_estimators_['xgb']
    st.sidebar.success("✅ Loaded enhanced model v3 (120 features)")
elif os.path.exists('ensemble_model_v2.pkl'):
    ensemble_model = joblib.load('ensemble_model_v2.pkl')
    feature_cols = joblib.load('feature_cols_v2.pkl')
    xgb_model = ensemble_model.named_estimators_['xgb']
    st.sidebar.success("✅ Loaded enhanced model v2")
else:
    st.sidebar.warning("Enhanced model not found, using original training")
    ensemble_model = None

if 'feature_cols' not in locals():
    feature_cols = []

# --- Load pooled model if available ---
pooled_model = None
pooled_feature_cols = None
if os.path.exists('pooled_model.pkl') and os.path.exists('pooled_feature_cols.pkl'):
    pooled_model = joblib.load('pooled_model.pkl')
    pooled_feature_cols = joblib.load('pooled_feature_cols.pkl')
    st.sidebar.success("✅ Loaded pooled model (trained on multiple stocks)")
else:
    st.sidebar.warning("Pooled model not found. Using per‑stock training (slower).")

# --- Load calibration map for pooled model ---
calibration_map = None
if os.path.exists('calibration_map.pkl'):
    calibration_map = joblib.load('calibration_map.pkl')
    st.sidebar.success("✅ Loaded probability calibration map")
else:
    st.sidebar.warning("Calibration map not found. Using raw probabilities.")

def calibrate_prob(prob, cal_map):
    if cal_map is None:
        return prob
    bins = cal_map['bin_edges']
    win_rates = cal_map['win_rates']
    for i in range(len(bins)-1):
        if bins[i] <= prob < bins[i+1]:
            return win_rates[i]
    if prob >= 1.0:
        return win_rates[-1] if win_rates else prob
    if prob < 0:
        return win_rates[0] if win_rates else prob
    return prob

# ------------------- Page config -------------------
st.set_page_config(page_title="AI Momentum Predictor", layout="wide")

# ------------------- AUTHENTICATION -------------------
if 'user_email' not in st.session_state:
    st.session_state.user_email = None

if not st.session_state.user_email:
    st.sidebar.title("🔐 Login / Sign Up")
    
    with st.sidebar.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
        if submitted:
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user_email = res.user.email
                # Save the session tokens
                st.session_state.supabase_session = {
                'access_token': res.session.access_token,
                'refresh_token': res.session.refresh_token
                }
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Login failed: {e}")

    st.sidebar.caption("🔐 Forgot your password? Email `stockpicksnathan316@gmail.com` and we'll help you reset it.")
    
    with st.sidebar.form("signup_form"):
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        st.caption("📧 **Please check your spam folder** for the confirmation email. Mark it as 'Not spam' so future emails land in your inbox.")
        signup_submitted = st.form_submit_button("Sign Up")
        if signup_submitted:
            try:
                # Sign up – no auto‑login
                supabase.auth.sign_up({"email": new_email, "password": new_password})
                st.sidebar.success("✅ Check your email for a confirmation link! You'll be able to log in after confirming.")
            except Exception as e:
                st.sidebar.error(f"Signup failed: {e}")
    
    # IMPORTANT: Stop the app here if not logged in
    st.stop()

# ------------------- After login: show user info & logout -------------------
st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")
if st.sidebar.button("Logout"):
    supabase.auth.sign_out()
    st.session_state.user_email = None
    st.session_state.supabase_session = None  # Clear saved tokens
    st.rerun()

# --- User Alert Dashboard (sidebar) ---
with st.sidebar.expander("🔔 My Alerts"):
    alerts = supabase.table("user_alerts").select("*").eq("user_email", st.session_state.user_email).execute()
    if alerts.data:
        for alert in alerts.data:
            col1, col2 = st.columns([3, 1])
            col1.write(f"{alert['ticker']} (α={alert['alpha']:.2f}, thresh={alert['threshold']:.2f})")
            if col2.button("🗑️", key=f"del_{alert['id']}"):
                supabase.table("user_alerts").delete().eq("id", alert['id']).execute()
                st.rerun()
    else:
        st.write("No alerts set. Click 'Watch' on a stock analysis to add one.")


# --- Hybrid model weight ---
alpha = st.sidebar.slider(
    "Hybrid weight (pooled model)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Weight given to the market‑wide pooled model vs. the stock‑specific model."
)
st.sidebar.caption(f"Final = {alpha:.0%} pooled + {1-alpha:.0%} stock‑specific")

# --- Trade threshold slider ---
class_threshold = st.sidebar.slider(
    "Trade threshold (min probability)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Signals with probability ≥ this value are considered. Lower = more trades."
)
st.sidebar.caption(f"Current threshold = {class_threshold:.2f}")

st.title("🤖 AI Momentum Predictor")
user = supabase.auth.get_user()
st.write("Authenticated user:", user.user.email if user and user.user else "None")


# ------------------- Stripe setup -------------------
stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]
price_id = st.secrets["stripe_price_id"]
base_url = st.secrets.get("base_url", "http://localhost:8501")

# Initialize paid_user in session state
if 'paid_user' not in st.session_state:
    st.session_state.paid_user = False

# Determine Pro status from database
st.session_state.paid_user = check_user_pro_status(st.session_state.user_email)

# ------------------- Handle Stripe return -------------------
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
            # Update paid_users table
            supabase.table("paid_users").update({"is_pro": True}).eq("email", st.session_state.user_email).execute()
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

# ------------------- License key input (optional) -------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Unlock Unlimited Scans")
license_key_input = st.sidebar.text_input("Enter license key", type="password")
if st.sidebar.button("Activate License"):
    valid_keys = st.secrets.get("license_keys", ["test123"])
    if license_key_input in valid_keys:
        # Activate Pro in database
        supabase.table("paid_users").update({"is_pro": True}).eq("email", st.session_state.user_email).execute()
        st.session_state.paid_user = True
        st.sidebar.success("License activated! You now have unlimited scans.")
        st.rerun()
    else:
        st.sidebar.error("Invalid license key")

# ------------------- Show remaining scans -------------------
if st.session_state.paid_user:
    st.sidebar.success("Premium subscriber - unlimited scans!")
else:
    scans_used = get_user_scans_used(st.session_state.user_email)
    remaining = max(0, 5 - scans_used)
    st.sidebar.info(f"Free tier: {remaining}/5 scans remaining")

# ------------------- Upgrade button -------------------
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
            customer_email=st.session_state.user_email
        )
        st.session_state.checkout_url = checkout_session.url
        st.markdown(f"👉 [Click here to complete payment]({checkout_session.url})")
    except Exception as e:
        st.error(f"❌ Error creating checkout session: {e}")

if "checkout_url" in st.session_state:
    url = st.session_state.checkout_url
    st.success("✅ Ready to subscribe! Click the button below to complete your payment.")
    st.link_button("💳 Pay $20/month and unlock unlimited scans", url)
    # Clear URL after showing button
    del st.session_state.checkout_url

# ------------------- MAIN APP (single ticker) -------------------
ticker = st.text_input("Stock", "AAPL", key="main_ticker")
period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="main_period")

# --- Single‑ticker scan button ---
scan_single = st.button("🔍 Run Analysis")

# Initialize session state for results
if 'single_ticker_results' not in st.session_state:
    st.session_state.single_ticker_results = None

from streamlit_autorefresh import st_autorefresh
st.sidebar.checkbox("Auto-refresh every 5 min", key="auto_refresh")
if st.session_state.auto_refresh:
    st_autorefresh(interval=300000, key="auto_refresh_timer")

if scan_single:
    # Check scan limit
    is_pro = st.session_state.paid_user
    if not is_pro:
        scans_used = get_user_scans_used(st.session_state.user_email)
        if scans_used >= 5:
            st.error("🔒 You've used all 5 free scans. Please upgrade to Pro for unlimited access!")
            st.stop()

    # ------------------- ANALYSIS BEGINS -------------------
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        st.error(f"❌ No data found for {ticker} for period {period}. Please try another ticker or period.")
        st.stop()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Debug: print raw data for AAPL
    if ticker == 'AAPL':
        print("\n--- Single‑ticker raw data (last 3 days) ---")
        print(df[['Open','High','Low','Close','Volume']].tail(3))

    df = safe_add_ta_features(df)

    # --- Get full macro + sector data ---
    macro_sector_df = get_macro_sector_data_cached(period)

    # --- Create a copy of the basic data (technicals + VIX, TNX, CL) for stock‑specific model ---
    basic_macro_df = macro_sector_df[['VIX', 'TNX', 'CL']]
    df_basic = df.copy()
    df_basic = df_basic.join(basic_macro_df, how='left').ffill().bfill()

    # ------------------------------------------------------------------
    # Branch based on alpha (hybrid weight)
    # ------------------------------------------------------------------
    if alpha == 0:
        # --- OLD PER‑STOCK SCANNER (technical indicators + basic macro only) ---
        # Use cached model
        model, feature_cols = get_stock_specific_model(ticker, df_basic)

        # Prepare latest row for live prediction
        latest_row = df_basic[feature_cols].fillna(0).iloc[[-1]]
        live_prob = model.predict_proba(latest_row)[0][1]

        # Prepare test set for backtest
        df_clean = df_basic.copy()
        df_clean['future_close'] = df_clean['Close'].shift(-5)
        df_clean['target'] = (df_clean['future_close'] > df_clean['Close']).astype(int)
        df_clean = df_clean.dropna(subset=['target']).copy()
        feature_columns = [col for col in df_clean.columns if col not in
                           ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']]
        split_idx = int(len(df_clean) * 0.8)
        X_test = df_clean.iloc[split_idx:][feature_columns].fillna(0)
        y_test = df_clean.iloc[split_idx:]['target']
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred_class = (y_test_pred_proba > class_threshold).astype(int)
        acc = accuracy_score(y_test, y_test_pred_class)
        prec = precision_score(y_test, y_test_pred_class, zero_division=0)

        xgb_best = model.named_estimators_['xgb']
        importance_features = feature_columns
        df_test = df_clean.iloc[split_idx:].copy()
        df_test['pred_prob'] = y_test_pred_proba
        df_test['pred_class'] = y_test_pred_class
        df_test['actual'] = y_test.values

        st.write("🎯 **Using old per‑stock scanner (cached)**")

        # Set consistent variables for result dict
        df_train = df_clean.iloc[:split_idx].copy()
        feature_columns = feature_columns
        X_test = X_test

    elif alpha == 1:
        # --- PURE POOLED MODEL (macro‑aware, with enhanced features) ---
        fundamentals = get_fundamentals(ticker)
        ticker_sector = TICKERS.get(ticker, 'Unknown')
        sector_etf = sector_to_etf.get(ticker_sector, None)
        df = fe.add_enhanced_features(df, ticker, macro_sector_df, sector_etf, fundamentals)

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
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        df_train = df_clean.iloc[:split_idx].copy()
        df_test = df_clean.iloc[split_idx:].copy()

        if pooled_model is not None and pooled_feature_cols is not None:
            X_test_aligned = X_test.reindex(columns=pooled_feature_cols, fill_value=0)
            y_test_pred_proba = pooled_model.predict_proba(X_test_aligned)[:, 1]
            y_test_pred_class = (y_test_pred_proba > class_threshold).astype(int)
            acc = accuracy_score(y_test, y_test_pred_class)
            prec = precision_score(y_test, y_test_pred_class, zero_division=0)

            latest_features = df_clean[feature_columns].iloc[[-1]].fillna(0).reindex(columns=pooled_feature_cols, fill_value=0)

            # Debug: print features for AAPL
            if ticker == 'AAPL':
                print(f"\n--- Single‑ticker features for AAPL ---")
                print(latest_features.head(10).T)
                print("----------------------------------------\n")

            live_prob_raw = pooled_model.predict_proba(latest_features)[0][1]
            live_prob_cal = calibrate_prob(live_prob_raw, calibration_map)

            xgb_best = pooled_model.named_estimators_['xgb']
            st.write("🎯 **Using pure pooled model (macro‑aware)**")
        else:
            st.error("Pooled model not loaded – cannot use alpha=1. Please retrain pooled model.")
            st.stop()

        importance_features = pooled_feature_cols

    else:
        # --- HYBRID MODEL (0 < alpha < 1) ---
        # 1. Pooled model prediction (enhanced features)
        fundamentals = get_fundamentals(ticker)
        ticker_sector = TICKERS.get(ticker, 'Unknown')
        sector_etf = sector_to_etf.get(ticker_sector, None)
        df_enhanced = fe.add_enhanced_features(df, ticker, macro_sector_df, sector_etf, fundamentals)

        df_enhanced['future_close'] = df_enhanced['Close'].shift(-5)
        df_enhanced['target'] = (df_enhanced['future_close'] > df_enhanced['Close']).astype(int)
        df_clean_enhanced = df_enhanced.dropna(subset=['target']).copy()
        feature_columns_enhanced = [col for col in df_clean_enhanced.columns if col not in
                                     ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']]

        X_enhanced = df_clean_enhanced[feature_columns_enhanced]
        y_enhanced = df_clean_enhanced['target']
        split_idx = int(len(df_clean_enhanced) * 0.8)
        X_train_enhanced = X_enhanced.iloc[:split_idx]
        X_test_enhanced = X_enhanced.iloc[split_idx:]
        y_test_enhanced = y_enhanced.iloc[split_idx:]

        X_test_aligned = X_test_enhanced.reindex(columns=pooled_feature_cols, fill_value=0)
        pooled_test_proba = pooled_model.predict_proba(X_test_aligned)[:, 1]

        latest_enhanced = df_clean_enhanced[feature_columns_enhanced].iloc[[-1]].fillna(0).reindex(columns=pooled_feature_cols, fill_value=0)
        pooled_live_prob = pooled_model.predict_proba(latest_enhanced)[0][1]

        # 2. Stock‑specific model (cached, using basic features)
        stock_model, stock_feature_cols = get_stock_specific_model(ticker, df_basic)

        # Prepare test set for stock model (using df_basic)
        df_clean_basic = df_basic.copy()
        df_clean_basic['future_close'] = df_clean_basic['Close'].shift(-5)
        df_clean_basic['target'] = (df_clean_basic['future_close'] > df_clean_basic['Close']).astype(int)
        df_clean_basic = df_clean_basic.dropna(subset=['target']).copy()
        feature_cols_basic = [col for col in df_clean_basic.columns if col not in
                              ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']]
        split_idx_basic = int(len(df_clean_basic) * 0.8)
        X_test_basic = df_clean_basic.iloc[split_idx_basic:][feature_cols_basic].fillna(0)
        y_test_basic = df_clean_basic.iloc[split_idx_basic:]['target']
        stock_test_proba = stock_model.predict_proba(X_test_basic)[:, 1]

        latest_basic = df_clean_basic[feature_cols_basic].fillna(0).iloc[[-1]]
        stock_live_prob = stock_model.predict_proba(latest_basic)[0][1]

        # 3. Blend probabilities
        y_test_pred_proba = alpha * pooled_test_proba + (1 - alpha) * stock_test_proba
        live_prob_raw = alpha * pooled_live_prob + (1 - alpha) * stock_live_prob
        live_prob_cal = calibrate_prob(live_prob_raw, calibration_map)

        # Use y_test from enhanced (same dates)
        y_test = y_test_enhanced
        y_test_pred_class = (y_test_pred_proba > class_threshold).astype(int)
        acc = accuracy_score(y_test, y_test_pred_class)
        prec = precision_score(y_test, y_test_pred_class, zero_division=0)

        # Build df_test for backtest
        df_test = df_clean_enhanced.iloc[split_idx:].copy()
        df_test['pred_prob'] = y_test_pred_proba
        df_test['pred_class'] = y_test_pred_class
        df_test['actual'] = y_test.values

        xgb_best = pooled_model.named_estimators_['xgb']
        importance_features = pooled_feature_cols

        st.write(f"🎯 **Using hybrid model ({alpha:.0%} pooled + {1-alpha:.0%} stock‑specific)**")

        # Set consistent variables for result dict
        df_train = df_clean_enhanced.iloc[:split_idx].copy()
        feature_columns = feature_columns_enhanced
        X_test = X_test_enhanced

    # --- Store all results in session state ---
    result_dict = {
        'df': df,
        'df_test': df_test,
        'acc': acc,
        'prec': prec,
        'y_test_pred_proba': y_test_pred_proba,
        'y_test_pred_class': y_test_pred_class,
        'y_test': y_test,
        'xgb_best': xgb_best,
        'importance_features': importance_features,
        'df_train': df_train,
        'feature_columns': feature_columns,
        'X_test': X_test,
    }

    # Add probability fields based on branch
    if alpha == 0:
        result_dict['live_prob'] = live_prob
    else:
        result_dict['live_prob_raw'] = live_prob_raw
        result_dict['live_prob_cal'] = live_prob_cal

    st.session_state.single_ticker_results = result_dict

    # --- Universal test‑set calibration for all branches (same as before) ---
    # (You can keep the existing calibration code here; it's unchanged)

    # --- Universal test‑set calibration for all branches ---
    # This computes the calibrated win rate from the test set
    # and stores it in res['live_prob_cal'].
    # It runs after the results are stored but before any UI display.

    res = st.session_state.single_ticker_results

    # Only compute if we have test data
    if len(res['X_test']) > 0:
        # Prepare test dataframe
        df_test = res['df_test'].copy()
        # Add predicted probabilities (stored in results)
        df_test['pred_prob'] = res['y_test_pred_proba']

        # Compute 5‑day forward return
        if 'future_close' not in df_test.columns:
            df_test['future_close'] = df_test['Close'].shift(-5)
        df_test['return_5d'] = (df_test['future_close'] - df_test['Close']) / df_test['Close']

        # Define probability bins
        bins = np.arange(0, 1.1, 0.1)
        labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]

        # Assign each test prediction to a bin
        df_test['prob_bin'] = pd.cut(df_test['pred_prob'], bins=bins, labels=labels, include_lowest=True)

        # Compute historical win rate per bin (5‑day return > 0)
        bin_win_rate = df_test.groupby('prob_bin')['return_5d'].apply(lambda x: (x > 0).mean()).fillna(0)

        # Current raw probability (depends on branch)
        if 'live_prob_raw' in res:
            current_prob = res['live_prob_raw']
        else:
            current_prob = res['live_prob']

        # Find which bin the current prediction falls into
        current_bin = pd.cut([current_prob], bins=bins, labels=labels, include_lowest=True)[0]
        if current_bin is not None:
            calibrated = bin_win_rate[current_bin]
        else:
            calibrated = None

        # Store the calibrated value in the results dictionary
        res['live_prob_cal'] = calibrated
        # Also store the full bin_win_rate for potential reuse (optional)
        res['calibration_bins'] = bin_win_rate
        res['calibration_labels'] = labels

        # Update session state
        st.session_state.single_ticker_results = res
        
    # Increment scan count if free user
    if not is_pro:
        increment_user_scans(st.session_state.user_email)

# ------------------------------------------------------------------
# Display results from session state (if any)
# ------------------------------------------------------------------
if st.session_state.single_ticker_results is not None:
    res = st.session_state.single_ticker_results

    # --- Candlestick chart ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=res['df'].index,
        open=res['df']['Open'],
        high=res['df']['High'],
        low=res['df']['Low'],
        close=res['df']['Close'],
        name='Price'
    ))

    if 'rsi' in res['df'].columns:
        buy_signals = res['df'][res['df']['rsi'] < 30]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'] * 0.98,
                mode='markers',
                marker=dict(size=8, color='lime', symbol='triangle-up'),
                name='RSI Buy Signal'
            ))

    st.plotly_chart(fig, width='stretch')

    # --- Metrics row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Test Accuracy", f"{res['acc']:.1%}")
    col2.metric("⚡ Test Precision", f"{res['prec']:.1%}")
    col3.metric("📊 Training Days", len(res['df_train']))

    col4, col5 = st.columns(2)
    if 'live_prob_raw' in res:
        # For alpha>0
        col4.metric("🔮 Raw 5‑Day UP Probability", f"{res['live_prob_raw']:.1%}")
        col5.metric("📊 Calibrated Win Rate", f"{res['live_prob_cal']:.1%}")
    else:
        # For alpha=0
        col4.metric("🔮 Raw 5‑Day UP Probability", f"{res['live_prob']:.1%}")
        if res.get('live_prob_cal') is not None:
            col5.metric("📊 Calibrated Win Rate", f"{res['live_prob_cal']:.1%}")
        else:
            col5.metric("📊 Calibrated Win Rate", "N/A")
    st.caption(f"📅 Latest Data: {str(res['df'].index[-1].date())}")

    # --- Suggested Position Size based on calibrated win rate ---
    if 'live_prob_cal' in res and res['live_prob_cal'] is not None:
        pos_size = res['live_prob_cal']
    elif st.session_state.get('calibration') and st.session_state.calibration.get('calibrated_prob') is not None:
        pos_size = st.session_state.calibration['calibrated_prob']
    else:
        pos_size = None

    if pos_size is not None:
        st.metric("💼 Suggested Position Size", f"{pos_size:.0%}")
    else:
        st.metric("💼 Suggested Position Size", "N/A")

    # --- Backtest on test set (out-of-sample) ---
    if len(res['X_test']) > 0:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc_test = accuracy_score(res['y_test'], res['y_test_pred_class'])
        prec_test = precision_score(res['y_test'], res['y_test_pred_class'], zero_division=0)
        rec_test = recall_score(res['y_test'], res['y_test_pred_class'], zero_division=0)
        f1_test = f1_score(res['y_test'], res['y_test_pred_class'], zero_division=0)

        # Store test predictions for plotting
        res['df_test']['pred_prob'] = res['y_test_pred_proba']
        res['df_test']['pred_class'] = res['y_test_pred_class']
        res['df_test']['actual'] = res['y_test'].values

        # --- Compute calibration for this ticker (before any expander) ---
        # Prepare df_test
        df_test = res['df_test'].copy()
        if 'future_close' not in df_test.columns:
            df_test['future_close'] = df_test['Close'].shift(-5)
        df_test['return_5d'] = (df_test['future_close'] - df_test['Close']) / df_test['Close']

        # Bins
        bins = np.arange(0, 1.1, 0.1)
        labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]
        df_test['prob_bin'] = pd.cut(df_test['pred_prob'], bins=bins, labels=labels, include_lowest=True)
        bin_win_rate = df_test.groupby('prob_bin')['return_5d'].apply(lambda x: (x > 0).mean()).fillna(0)

        # Current prediction
        if 'live_prob_raw' in res:
            current_prob = res['live_prob_raw']
        else:
            current_prob = res['live_prob']

        current_bin = pd.cut([current_prob], bins=bins, labels=labels, include_lowest=True)[0]
        if current_bin is not None:
            win_rate_for_bin = bin_win_rate[current_bin]
            # Add this to res so metrics row can use it
            res['live_prob_cal'] = win_rate_for_bin
        else:
            res['live_prob_cal'] = None

        # --- Backtest expander ---
        with st.expander("📈 Backtest Performance (out‑of‑sample)"):
            st.metric("Test Accuracy", f"{acc_test:.1%}")
            st.metric("Test Precision", f"{prec_test:.1%}")
            st.metric("Test Recall", f"{rec_test:.1%}")
            st.metric("Test F1 Score", f"{f1_test:.1%}")

            # Prepare test dataframe with 5‑day return
            df_test = res['df_test'].copy()
            if 'future_close' not in df_test.columns:
                df_test['future_close'] = df_test['Close'].shift(-5)
            df_test['return_5d'] = (df_test['future_close'] - df_test['Close']) / df_test['Close']

            # Use the same classification threshold set earlier
            threshold = class_threshold
            st.write(f"Max predicted probability: {df_test['pred_prob'].max():.3f}")

            # Generate signals: predicted probability > threshold
            signals = df_test[df_test['pred_prob'] > threshold].copy()
            if len(signals) == 0:
                st.info("No trades were taken during the test period.")
            else:
                signals['entry_date'] = signals.index
                # Function to compute 5‑day forward return for a given date
                def get_future_return(row_idx, df):
                    try:
                        pos = df.index.get_loc(row_idx)
                        if pos + 5 < len(df):
                            future_idx = df.index[pos + 5]
                            future_price = df.loc[future_idx, 'Close']
                            current_price = df.loc[row_idx, 'Close']
                            return (future_price - current_price) / current_price
                        else:
                            return np.nan
                    except Exception:
                        return np.nan

                signals['return_5d'] = signals['entry_date'].apply(lambda x: get_future_return(x, df_test))
                # Force numeric and drop rows where conversion fails (non‑numeric, NaNs)
                signals['return_5d'] = pd.to_numeric(signals['return_5d'], errors='coerce')
                signals = signals.dropna(subset=['return_5d']).copy()
                signals['win'] = signals['return_5d'] > 0
                signals['exit_date'] = signals['entry_date'] + pd.Timedelta(days=5*5/7)  # approximate 5 trading days

                # --- Trade‑by‑trade table ---
                trade_table = pd.DataFrame({
                    'Entry Date': signals['entry_date'].dt.strftime('%Y-%m-%d'),
                    'Exit Date': signals['exit_date'].dt.strftime('%Y-%m-%d'),
                    'Predicted Probability': signals['pred_prob'].round(3),
                    '5‑Day Return': signals['return_5d'].round(4),
                    'Win/Loss': signals['win'].map({True: 'Win', False: 'Loss'})
                })
                st.subheader("📊 Trade-by-Trade Breakdown (5-Day Hold)")
                st.dataframe(trade_table, use_container_width=True)

                # Summary metrics
                win_rate = signals['win'].mean()
                avg_win = signals.loc[signals['win'], 'return_5d'].mean() if signals['win'].any() else 0
                avg_loss = signals.loc[~signals['win'], 'return_5d'].mean() if (~signals['win']).any() else 0
                profit_factor = (signals.loc[signals['win'], 'return_5d'].sum() / abs(signals.loc[~signals['win'], 'return_5d'].sum())) if (~signals['win']).any() and signals.loc[signals['win'], 'return_5d'].sum() > 0 else np.inf

                col1, col2, col3 = st.columns(3)
                col1.metric("Win Rate", f"{win_rate:.1%}")
                col2.metric("Avg Win", f"{avg_win:.2%}")
                col2.metric("Avg Loss", f"{avg_loss:.2%}")
                col3.metric("Profit Factor", f"{profit_factor:.2f}")

                # --- Equity curve (simplified) ---
                # Create a series of zeros with the same index as df_test
                strategy_returns = pd.Series(0.0, index=df_test.index)
                for _, row in signals.iterrows():
                    entry = row['entry_date']
                    ret = row['return_5d']
                    pos = df_test.index.get_loc(entry)
                    exit_pos = min(pos + 5, len(df_test)-1)
                    exit_date = df_test.index[exit_pos]
                    strategy_returns[exit_date] += ret

                strategy_cumulative = (1 + strategy_returns).cumprod()
                market_cumulative = (1 + df_test['return_5d'].fillna(0)).cumprod()

                fig_back = go.Figure()
                fig_back.add_trace(go.Scatter(x=df_test.index, y=market_cumulative,
                                              mode='lines', name='Buy & Hold'))
                fig_back.add_trace(go.Scatter(x=strategy_cumulative.index, y=strategy_cumulative,
                                              mode='lines', name='AI Strategy (5‑day hold)'))
                fig_back.update_layout(title="Equity Curve (Test Period)", xaxis_title="Date", yaxis_title="Cumulative Return")
                st.plotly_chart(fig_back, use_container_width=True)

            # --- Probability Calibration (nested expander) ---
            with st.expander("📊 Probability Calibration"):
                # Define probability bins
                bins = np.arange(0, 1.1, 0.1)
                labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]

                # Ensure we have the 5‑day return column
                if 'return_5d' not in df_test.columns:
                    df_test['future_close'] = df_test['Close'].shift(-5)
                    df_test['return_5d'] = (df_test['future_close'] - df_test['Close']) / df_test['Close']

                df_test['prob_bin'] = pd.cut(df_test['pred_prob'], bins=bins, labels=labels, include_lowest=True)
                bin_win_rate = df_test.groupby('prob_bin')['return_5d'].apply(lambda x: (x > 0).mean()).fillna(0)
                bin_count = df_test.groupby('prob_bin')['return_5d'].count()

                cal_df = pd.DataFrame({
                    'Probability Bin': bin_win_rate.index,
                    'Number of Trades': bin_count.values,
                    'Historical Win Rate': bin_win_rate.values
                }).reset_index(drop=True)

                st.dataframe(cal_df.style.format({'Historical Win Rate': '{:.1%}'}), use_container_width=True)

                # Find bin for current prediction
                if 'live_prob_raw' in res:
                    current_prob = res['live_prob_raw']
                else:
                    current_prob = res['live_prob']

                current_bin = pd.cut([current_prob], bins=bins, labels=labels, include_lowest=True)[0]
                if current_bin is not None:
                    win_rate_for_bin = bin_win_rate[current_bin]
                    st.info(
                        f"📊 **Current prediction ({current_prob:.1%})** falls into bin **{current_bin}**.\n\n"
                        f"Historically, signals in this bin had a **win rate of {win_rate_for_bin:.1%}**."
                    )
                    if alpha == 0:
                        st.session_state.calibration = {
                            'bins': bins,
                            'win_rates': bin_win_rate.values,
                            'current_bin': current_bin,
                            'calibrated_prob': win_rate_for_bin
                        }
                else:
                    st.info(f"Current probability {current_prob:.1%} is outside the bins.")

                # Reliability diagram
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Bar(x=cal_df['Probability Bin'], y=cal_df['Historical Win Rate'], name='Actual Win Rate'))
                fig_cal.update_layout(title='Probability Calibration (5‑Day Return)', xaxis_title='Predicted Probability Bin', yaxis_title='Actual Win Rate')
                st.plotly_chart(fig_cal, use_container_width=True)

    else:
        with st.expander("📈 Backtest Performance (out‑of‑sample)"):
            st.info("Not enough test data to display backtest.")

    # --- Feature importance (optional) ---
    if st.checkbox("Show what XGBoost learned"):
        importance = pd.DataFrame({
            'feature': res['importance_features'],
            'importance': res['xgb_best'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        import plotly.express as px
        fig = px.bar(importance,
                     x='importance',
                     y='feature',
                     orientation='h',
                     title='Top 10 Feature Importances',
                     labels={'importance': 'Importance', 'feature': ''})
        fig.update_layout(yaxis={'categoryorder':'total ascending'},
                          height=400,
                          margin=dict(l=150))
        st.plotly_chart(fig, use_container_width=True)

    # --- Watch this stock button ---
    if st.button("🔔 Watch this stock"):
        # Determine model_type based on alpha
        if alpha == 0:
            model_type = "old"
        elif alpha == 1:
            model_type = "pooled"
        else:
            model_type = "hybrid"

        supabase.table("user_alerts").upsert({
            "user_email": st.session_state.user_email,
            "ticker": ticker,
            "alpha": alpha,
            "threshold": class_threshold,
            "model_type": model_type
        }, on_conflict="user_email, ticker").execute()
        st.success("Alert added! You'll receive an email when the probability meets your threshold.")
        st.rerun()   # Force a rerun to update the dashboard immediately

else:
    st.info("Click 'Run Analysis' to generate predictions and backtest.")


# ------------------- MULTI‑TICKER SCREENER -------------------
st.sidebar.header("🔍 Market Scanner (50+ Tickers)")

sectors = ['All'] + sorted(set(TICKERS.values()))
selected_sector = st.sidebar.selectbox("Filter by sector", sectors)

# New: period selector for the multi‑scanner
multi_period = st.sidebar.selectbox("Scanner period", ["1y", "2y", "6mo"], index=0, key="multi_period")

# --- In the multi‑scanner sidebar, after period selector ---
use_hybrid = st.sidebar.checkbox("Use hybrid model (slower, more accurate)", value=False)
if alpha == 0:
    st.sidebar.info("Hybrid mode disabled when weight=0 (old scanner)")
    use_hybrid = False

if st.sidebar.button("🗑️ Clear Scanner Cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Rerun scans to refresh.")

scan_button = st.sidebar.button("Scan Selected Tickers")

if selected_sector == 'All':
    ticker_list = list(TICKERS.keys())
else:
    ticker_list = [t for t, s in TICKERS.items() if s == selected_sector]

st.sidebar.write(f"📊 Scanning **{len(ticker_list)}** tickers")

@st.cache_data(ttl=21600)
def scan_tickers_fallback(tickers, macro_sector_df, alpha, period, use_hybrid):
    print(f"[CACHE] Alpha = {alpha}")
    results = []
    for ticker in tickers:
        try:
            df_t = yf.download(ticker, period=period, progress=False)
            if df_t.empty:
                results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": "No data", "Prob": 0.0})
                continue

            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.droplevel(1)

            # Debug: print raw data for AAPL
            if ticker == 'AAPL':
                print("\n--- Multi‑scanner raw data (last 3 days) ---")
                print(df_t[['Open','High','Low','Close','Volume']].tail(3))

            df_t = safe_add_ta_features(df_t)

            # --- Basic DataFrame for stock‑specific model (technical indicators + VIX, TNX, CL) ---
            basic_macro = macro_sector_df[['VIX', 'TNX', 'CL']]
            df_t_basic = df_t.copy()
            df_t_basic = df_t_basic.join(basic_macro, how='left').ffill().bfill()

            if alpha < 0.01:          # treat values less than 0.01 as zero
                print("[CACHE] Using old per-stock model")
                
                # --- OLD PER‑STOCK SCANNER (no enhanced features) ---
                feature_cols = [c for c in df_t_basic.columns if c not in ['Open','High','Low','Close','Volume']]
                split = int(len(df_t_basic) * 0.8)
                train = df_t_basic.iloc[:split]
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

                latest = df_t_basic[feature_cols].fillna(0).iloc[[-1]]
                prob = ensemble_t.predict_proba(latest)[0][1]
                results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": f"{prob:.1%}", "Prob": prob})

            else:
                print("[CACHE] Using pooled model")

                # --- Enhanced DataFrame for pooled model (sector ETFs, relative strength, fundamentals) ---
                fundamentals = get_fundamentals(ticker)
                ticker_sector = TICKERS.get(ticker, 'Unknown')
                sector_etf = sector_to_etf.get(ticker_sector, None)
                df_t_enhanced = fe.add_enhanced_features(df_t.copy(), ticker, macro_sector_df, sector_etf, fundamentals)

                if pooled_model is not None and pooled_feature_cols is not None:
                    # Align columns for pooled model
                    for col in pooled_feature_cols:
                        if col not in df_t_enhanced.columns:
                            df_t_enhanced[col] = 0
                    latest_enhanced = df_t_enhanced[pooled_feature_cols].fillna(0).iloc[[-1]]

                    # Debug: print features for AAPL
                    if ticker == 'AAPL':
                        print(f"\n--- Multi‑scanner features for AAPL ---")
                        print(latest_enhanced.head(10).T)
                        print("---------------------------------------\n")

                    prob_raw = pooled_model.predict_proba(latest_enhanced)[0][1]

                    # --- Hybrid blending (if enabled and alpha between 0 and 1) ---
                    if use_hybrid and 0 < alpha < 1:
                        # Get cached stock‑specific model (using basic data)
                        stock_model, stock_feature_cols = get_stock_specific_model(ticker, df_t_basic)
                        latest_basic = df_t_basic[stock_feature_cols].fillna(0).iloc[[-1]]
                        stock_prob = stock_model.predict_proba(latest_basic)[0][1]
                        prob_raw = alpha * prob_raw + (1 - alpha) * stock_prob

                else:
                    # Fallback (should not happen)
                    feature_cols = [c for c in df_t_basic.columns if c not in ['Open','High','Low','Close','Volume']]
                    split = int(len(df_t_basic) * 0.8)
                    train = df_t_basic.iloc[:split]
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

                    latest = df_t_basic[feature_cols].fillna(0).iloc[[-1]]
                    prob_raw = ensemble_t.predict_proba(latest)[0][1]

                prob_cal = calibrate_prob(prob_raw, calibration_map)

                # Debug: print for a few tickers
                if ticker in ['AAPL', 'MSFT', 'NVDA']:
                    print(f"[DEBUG] {ticker}: raw={prob_raw:.3f}, cal={prob_cal:.3f}")

                results.append({
                    "Ticker": ticker,
                    "Sector": TICKERS.get(ticker, "Unknown"),
                    "Signal": f"{prob_raw:.1%}",
                    "Prob": prob_raw,
                    "Calibrated": prob_cal,
                    "Position": f"{prob_cal:.0%}"
                })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": "Error", "Prob": 0.0})

    return results



if scan_button:
    user_email = st.session_state.user_email
    is_pro = st.session_state.paid_user

    if not is_pro:
        scans_used = get_user_scans_used(user_email)
        if scans_used >= 5:
            st.sidebar.error("🔒 You've used all 5 free scans. Please upgrade to Pro for unlimited access!")
            st.stop()
        else:
            with st.spinner(f"Scanning {len(ticker_list)} tickers... this may take a minute."):
                macro_sector_df = get_macro_sector_data_cached(multi_period)
                results = scan_tickers_fallback(ticker_list, macro_sector_df, alpha, multi_period, use_hybrid)
                increment_user_scans(user_email)
                st.session_state.scanner_results = results
                st.rerun()
    else:
        with st.spinner(f"Scanning {len(ticker_list)} tickers... this may take a minute."):
            macro_sector_df = get_macro_sector_data_cached(multi_period)
            results = scan_tickers_fallback(ticker_list, macro_sector_df, alpha, multi_period, use_hybrid)
            st.session_state.scanner_results = results
            st.rerun()

# ------------------- Display scanner results (unchanged) -------------------
if st.session_state.get('scanner_results'):
    results = st.session_state.scanner_results
    df_results = pd.DataFrame(results)
    if 'Calibrated' in df_results.columns:
        df_results['Calibrated'] = pd.to_numeric(df_results['Calibrated'], errors='coerce')
        df_results = df_results.sort_values('Calibrated', ascending=False)
        # Keep 'Position' if present, drop the other intermediate columns
        cols_to_drop = ['Prob', 'Calibrated']
        if 'Position' in df_results.columns:
            df_results = df_results.drop(columns=[c for c in cols_to_drop if c in df_results.columns])
        else:
            df_results = df_results.drop(columns=cols_to_drop)
    else:
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

    with col_bear:
        st.error("🥶 **Most Bearish**")
        if bearish is not None:
            st.metric(bearish['Ticker'], bearish['Signal'], delta=None)
            st.caption(f"Sector: {bearish['Sector']}")

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