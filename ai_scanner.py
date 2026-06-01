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
import time

def ensure_numeric(df):
    """Convert all columns to float32, replace inf, and fill NaN with 0."""
    df = df.astype(np.float32)
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)
    return df

MIN_TRADES_FOR_CALIBRATION = 10

# ------------------- Calibration helper -------------------
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

# ------------------- Profit target model loader -------------------
def get_model_for_target(target_str):
    mapping = {
        "Any up": ("pooled_model_any_up.pkl", "calibration_map_any_up.pkl", "pooled_feature_cols_any_up.pkl"),
        "2%":    ("pooled_model_2pct.pkl", "calibration_map_2pct.pkl", "pooled_feature_cols_2pct.pkl"),
        "3%":    ("pooled_model_3pct.pkl", "calibration_map_3pct.pkl", "pooled_feature_cols_3pct.pkl"),
        "5%":    ("pooled_model_5pct.pkl", "calibration_map_5pct.pkl", "pooled_feature_cols_5pct.pkl"),
    }
    model_file, cal_file, feat_file = mapping.get(target_str, mapping["Any up"])
    
    # Fallback to existing models if new ones don't exist yet
    if not os.path.exists(model_file) and target_str != "Any up":
        st.warning(f"Model for {target_str} not found. Falling back to 'Any up'.")
        return get_model_for_target("Any up")
    
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        cal_map = joblib.load(cal_file)
        feat_cols = joblib.load(feat_file)
        return model, cal_map, feat_cols
    else:
        # Use your original pooled model as ultimate fallback
        return joblib.load('pooled_model.pkl'), joblib.load('calibration_map.pkl'), joblib.load('pooled_feature_cols.pkl')

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

def is_above_long_term_ma(df, period=200):
    """Return True if the latest close is above the period-day SMA."""
    if len(df) < period:
        return True   # Not enough data – allow trade (or you could return False)
    sma = df['Close'].rolling(window=period).mean()
    return df['Close'].iloc[-1] > sma.iloc[-1]

# --- Stock‑specific model caching ---
def get_stock_model_cache_path(ticker):
    ticker_hash = hashlib.md5(ticker.encode()).hexdigest()
    cache_dir = "tick_sniper_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    model_path = os.path.join(cache_dir, f"stock_model_{ticker_hash}.pkl")
    features_path = os.path.join(cache_dir, f"stock_features_{ticker_hash}.pkl")
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
    st.session_state.supabase_session = None
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

# ------------------- NEW: Profit target selector (MUST be before model loading) -------------------
profit_target = st.sidebar.selectbox(
    "Profit target for prediction",
    options=["Any up", "2%", "3%", "5%"],
    index=0,
    help="Model predicts probability of at least this gain in 5 days"
)

# ------------------- Load pooled model based on selected profit target -------------------
pooled_model = None
pooled_feature_cols = None
calibration_map = None

# Helper function (if not already defined above – it is, but we keep it here for clarity)
def get_model_for_target(target_str):
    mapping = {
        "Any up": ("pooled_model_any_up.pkl", "calibration_map_any_up.pkl", "pooled_feature_cols_any_up.pkl"),
        "2%":    ("pooled_model_2pct.pkl", "calibration_map_2pct.pkl", "pooled_feature_cols_2pct.pkl"),
        "3%":    ("pooled_model_3pct.pkl", "calibration_map_3pct.pkl", "pooled_feature_cols_3pct.pkl"),
        "5%":    ("pooled_model_5pct.pkl", "calibration_map_5pct.pkl", "pooled_feature_cols_5pct.pkl"),
    }
    model_file, cal_file, feat_file = mapping.get(target_str, mapping["Any up"])
    if os.path.exists(model_file):
        return joblib.load(model_file), joblib.load(cal_file), joblib.load(feat_file)
    else:
        st.warning(f"Model for {target_str} not found. Falling back to 'Any up'.")
        # Fallback to original pooled model
        if os.path.exists('pooled_model.pkl'):
            return joblib.load('pooled_model.pkl'), joblib.load('calibration_map.pkl'), joblib.load('pooled_feature_cols.pkl')
        else:
            return None, None, None

if profit_target != "Any up":
    # Try to load target-specific model
    pm, cm, pf = get_model_for_target(profit_target)
    if pm is not None:
        pooled_model, calibration_map, pooled_feature_cols = pm, cm, pf
        st.sidebar.success(f"✅ Loaded {profit_target} pooled model")
    else:
        st.sidebar.error(f"Could not load model for {profit_target}. Please train it first.")
        profit_target = "Any up"  # fallback for UI display
        # Then load default model
        if os.path.exists('pooled_model.pkl'):
            pooled_model = joblib.load('pooled_model.pkl')
            calibration_map = joblib.load('calibration_map.pkl')
            pooled_feature_cols = joblib.load('pooled_feature_cols.pkl')
            st.sidebar.success("✅ Loaded default pooled model (any up move)")
        else:
            pooled_model = None
            st.sidebar.warning("Pooled model not found. Using per‑stock training (slower).")
else:
    # "Any up" selected – load original pooled model
    if os.path.exists('pooled_model.pkl'):
        pooled_model = joblib.load('pooled_model.pkl')
        calibration_map = joblib.load('calibration_map.pkl')
        pooled_feature_cols = joblib.load('pooled_feature_cols.pkl')
        st.sidebar.success("✅ Loaded pooled model (any up move)")
    else:
        pooled_model = None
        pooled_feature_cols = None
        st.sidebar.warning("Pooled model not found. Using per‑stock training (slower).")

# --- FIX: Determine xgb_best once (works for both VotingClassifier and single XGBoost) ---
if pooled_model is not None:
    if hasattr(pooled_model, 'named_estimators_'):
        xgb_best_global = pooled_model.named_estimators_['xgb']   # VotingClassifier ensemble
    else:
        xgb_best_global = pooled_model                             # plain XGBClassifier
else:
    xgb_best_global = None


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


# --- NEW: Take profit & stop loss sliders ---
st.sidebar.markdown("### 🎯 Exit rules (ATR‑based)")
tp_atr_mult = st.sidebar.number_input(
    "Take profit (× ATR)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
    help="Exit when price rises by this multiple of ATR"
)
sl_atr_mult = st.sidebar.number_input(
    "Stop loss (× ATR)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
    help="Exit when price falls by this multiple of ATR"
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
        # --- OLD PER‑STOCK SCANNER ---
        model, feature_cols = get_stock_specific_model(ticker, df_basic)
        latest_row = df_basic[feature_cols].fillna(0).iloc[[-1]]
        live_prob = model.predict_proba(latest_row)[0][1]

        # Trend filter
        if not is_above_long_term_ma(df, period=200):
            st.warning("⚠️ Stock is below its 200-day moving average. Long signals are filtered out.")
            live_prob = 0.0

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
            X_test_aligned = ensure_numeric(X_test_aligned)
            y_test_pred_proba = pooled_model.predict_proba(X_test_aligned)[:, 1]
            y_test_pred_class = (y_test_pred_proba > class_threshold).astype(int)
            acc = accuracy_score(y_test, y_test_pred_class)
            prec = precision_score(y_test, y_test_pred_class, zero_division=0)

            latest_features = df_clean[feature_columns].iloc[[-1]].fillna(0).reindex(columns=pooled_feature_cols, fill_value=0)
            latest_features = ensure_numeric(latest_features)


            live_prob_raw = pooled_model.predict_proba(latest_features)[0][1]
            live_prob_cal = calibrate_prob(live_prob_raw, calibration_map)

            # Trend filter
            if not is_above_long_term_ma(df, period=200):
                st.warning("⚠️ Stock is below its 200-day moving average. Long signals are filtered out.")
                live_prob_raw = 0.0
                live_prob_cal = 0.0

            xgb_best = xgb_best_global
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
        # Convert all columns to float32 (XGBoost requires numeric input)
        X_test_aligned = X_test_aligned.astype(np.float32)
        pooled_test_proba = pooled_model.predict_proba(X_test_aligned)[:, 1]

        latest_enhanced = df_clean_enhanced[feature_columns_enhanced].iloc[[-1]].fillna(0).reindex(columns=pooled_feature_cols, fill_value=0)
        latest_enhanced = ensure_numeric(latest_enhanced)
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
        X_test_basic = ensure_numeric(X_test_basic) 
        y_test_basic = df_clean_basic.iloc[split_idx_basic:]['target']
        stock_test_proba = stock_model.predict_proba(X_test_basic)[:, 1]

        latest_basic = df_clean_basic[feature_cols_basic].fillna(0).iloc[[-1]]
        latest_basic = ensure_numeric(latest_basic)
        stock_live_prob = stock_model.predict_proba(latest_basic)[0][1]

        # 3. Blend probabilities
        y_test_pred_proba = alpha * pooled_test_proba + (1 - alpha) * stock_test_proba
        live_prob_raw = alpha * pooled_live_prob + (1 - alpha) * stock_live_prob
        live_prob_cal = calibrate_prob(live_prob_raw, calibration_map)

        # Trend filter
        if not is_above_long_term_ma(df, period=200):
            st.warning("⚠️ Stock is below its 200-day moving average. Long signals are filtered out.")
            live_prob_raw = 0.0
            live_prob_cal = 0.0

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

        importance_features = pooled_feature_cols

        st.write(f"🎯 **Using hybrid model ({alpha:.0%} pooled + {1-alpha:.0%} stock‑specific)**")
        xgb_best = xgb_best_global

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
        bin_count = df_test.groupby('prob_bin')['return_5d'].count()   # <-- add this line

        # Current raw probability (depends on branch)
        if 'live_prob_raw' in res:
            current_prob = res['live_prob_raw']
        else:
            current_prob = res['live_prob']

        # Find which bin the current prediction falls into
        current_bin = pd.cut([current_prob], bins=bins, labels=labels, include_lowest=True)[0]
        if current_bin is not None:
            bin_trades = bin_count[current_bin]
            if bin_trades >= MIN_TRADES_FOR_CALIBRATION:
                calibrated = bin_win_rate[current_bin]
            else:
                calibrated = current_prob
                st.warning(f"⚠️ Only {bin_trades} trades in bin {current_bin}. Using raw probability ({current_prob:.1%}) instead of calibrated win rate.")
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
        col4.metric(f"🔮 Prob. of ≥{profit_target} in 5d", f"{res['live_prob_raw']:.1%}")
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

            # --- NEW backtest with TP/SL ---
            threshold = class_threshold
            # Calculate ATR (14-day) for the test DataFrame
            df_test['tr'] = np.maximum(
                df_test['High'] - df_test['Low'],
                np.maximum(
                    abs(df_test['High'] - df_test['Close'].shift(1)),
                    abs(df_test['Low'] - df_test['Close'].shift(1))
                )
            )
            df_test['atr'] = df_test['tr'].rolling(14).mean()

            signals = df_test[df_test['pred_prob'] > threshold].copy()
            if len(signals) == 0:
                st.info("No trades were taken during the test period.")
            else:
                trade_list = []
                for idx, row in signals.iterrows():
                    entry_date = idx
                    entry_price = row['Close']
                    # Get ATR at entry (use previous day's ATR to avoid look-ahead)
                    entry_pos = df_test.index.get_loc(entry_date)
                    atr_entry = df_test['atr'].iloc[entry_pos] if not pd.isna(df_test['atr'].iloc[entry_pos]) else df_test['atr'].dropna().iloc[-1]
                    tp_price = entry_price * (1 + tp_atr_mult * (atr_entry / entry_price))
                    sl_price = entry_price * (1 - sl_atr_mult * (atr_entry / entry_price))
        
                    # Find the position of entry date in df_test
                    entry_pos = df_test.index.get_loc(entry_date)
                    exit_date = None
                    exit_price = None
                    exit_reason = None
                    
                    # Look at the next 5 trading days (or until we hit TP/SL)
                    for offset in range(1, 6):
                        if entry_pos + offset >= len(df_test):
                            break
                        day = df_test.iloc[entry_pos + offset]
                        day_high = day['High']
                        day_low = day['Low']
                
                        # Check stop loss first (more conservative)
                        if day_low <= sl_price:
                            exit_date = day.name
                            exit_price = sl_price
                            exit_reason = "Stop loss"
                            break
                        # Check take profit
                        if day_high >= tp_price:
                            exit_date = day.name
                            exit_price = tp_price
                            exit_reason = "Take profit"
                            break
            
                    # If neither hit within 5 days, close at the 5th day's close
                    if exit_date is None and entry_pos + 5 < len(df_test):
                        exit_date = df_test.index[entry_pos + 5]
                        exit_price = df_test.iloc[entry_pos + 5]['Close']
                        exit_reason = "Hold 5 days"
                    elif exit_date is None:
                        # Not enough data to hold 5 days – skip this trade
                        continue
            
                    ret = (exit_price - entry_price) / entry_price
                    trade_list.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': ret,
                        'win': ret > 0,
                        'reason': exit_reason,
                        'pred_prob': row['pred_prob']
                    })
        
                if not trade_list:
                    st.info("No trades completed (insufficient data after entry).")
                else:
                    trades_df = pd.DataFrame(trade_list)
            
                    # --- Trade table ---
                    trade_table = pd.DataFrame({
                        'Entry Date': trades_df['entry_date'].dt.strftime('%Y-%m-%d'),
                        'Exit Date': trades_df['exit_date'].dt.strftime('%Y-%m-%d'),
                        'Exit Reason': trades_df['reason'],
                        'Predicted Prob': trades_df['pred_prob'].round(3),
                        'Return': trades_df['return'].round(4),
                        'Win/Loss': trades_df['win'].map({True: 'Win', False: 'Loss'})
                    })
                    st.subheader("📊 Trade-by-Trade Breakdown (TP/SL exit)")
                    st.dataframe(trade_table, use_container_width=True)
            
                    # --- Summary metrics ---
                    win_rate = trades_df['win'].mean()
                    avg_win = trades_df.loc[trades_df['win'], 'return'].mean() if trades_df['win'].any() else 0
                    avg_loss = trades_df.loc[~trades_df['win'], 'return'].mean() if (~trades_df['win']).any() else 0
                    profit_factor = (trades_df.loc[trades_df['win'], 'return'].sum() / 
                                     abs(trades_df.loc[~trades_df['win'], 'return'].sum())) if (~trades_df['win']).any() and trades_df.loc[trades_df['win'], 'return'].sum() > 0 else np.inf
            
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Win Rate", f"{win_rate:.1%}")
                    col2.metric("Avg Win", f"{avg_win:.2%}")
                    col2.metric("Avg Loss", f"{avg_loss:.2%}")
                    col3.metric("Profit Factor", f"{profit_factor:.2f}")
                
                    # --- Exit reason breakdown (optional) ---
                    st.caption("Exit reason distribution")
                    reason_counts = trades_df['reason'].value_counts()
                    st.bar_chart(reason_counts)
            
                    # --- Equity curve (using actual exit dates) ---
                    # We'll build a daily return series that applies each trade's return on its exit date
                    strategy_returns = pd.Series(0.0, index=df_test.index)
                    for _, trade in trades_df.iterrows():
                        exit_date = trade['exit_date']
                        # If multiple trades exit on the same day, sum their returns
                        strategy_returns[exit_date] += trade['return']
        
                    strategy_cumulative = (1 + strategy_returns).cumprod()
                    # Benchmark: buy & hold on the same days? For simplicity, use daily returns of the stock
                    market_returns = df_test['Close'].pct_change().fillna(0)
                    market_cumulative = (1 + market_returns).cumprod()
        
                    fig_back = go.Figure()
                    fig_back.add_trace(go.Scatter(x=df_test.index, y=market_cumulative,
                                                  mode='lines', name='Buy & Hold'))
                    fig_back.add_trace(go.Scatter(x=strategy_cumulative.index, y=strategy_cumulative,
                                                  mode='lines', name='AI Strategy (TP/SL)'))
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

    # ==================== DURABILITY ASSESSMENT ====================
    with st.expander("📊 Full Stock Health Assessment (Durable Competitive Advantage)"):
        from durability_advanced import get_advanced_durability
        ticker = st.session_state.main_ticker
        adv = get_advanced_durability(ticker)
        if adv and adv['grade'] != 'Insufficient Data':
            grade_color = {'A':'green','B':'lightgreen','C':'orange','D':'salmon','F':'red'}.get(adv['grade'], 'gray')
            st.markdown(f"### Overall Durable Competitive Advantage Grade: <span style='background-color:{grade_color}; padding:0.2em 0.5em; border-radius:0.3em; font-size:1.5em;'>{adv['grade']}</span>", unsafe_allow_html=True)
            st.caption(f"Tally: ✅ {adv['strong_count']} strong signs, ⚠️ {adv['moderate_count']} moderate, ❌ {adv['weak_count']} weak")
            
            # Show ALL metrics in a clean table
            with st.expander("📋 Detailed Assessment Results (all metrics)", expanded=True):
                df_details = pd.DataFrame(list(adv['details'].items()), columns=['Metric', 'Assessment'])
                st.dataframe(df_details, use_container_width=True, height=400)
            
            # Optional: show raw numeric metrics in a second expander
            with st.expander("🔢 Raw Numerical Metrics"):
                num_metrics = {k: v for k, v in adv['metrics'].items() if not isinstance(v, pd.Series)}
                df_num = pd.DataFrame(list(num_metrics.items()), columns=['Metric', 'Value'])
                st.dataframe(df_num)
        else:
            st.info(f"Insufficient financial data for {ticker}. Check that ticker symbol is correct and the company has at least 5 years of annual financial statements.")

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
def scan_tickers_fallback(tickers, macro_sector_df, alpha, period, use_hybrid, profit_target):
    """
    Scans a list of tickers using batch download for price data (1 API call for all tickers).
    Then processes each ticker sequentially with the existing model logic.
    """
    results = []
    
    # ---------------------------
    # 1. Batch download all tickers' price data in a single request
    # ---------------------------
    try:
        # Download all tickers at once, group by ticker
        all_data = yf.download(tickers, period=period, group_by='ticker', progress=False)
        if all_data.empty:
            st.error("Batch download returned no data.")
            return results
    except Exception as e:
        st.error(f"Batch download failed: {e}")
        return results

    # If only one ticker was passed, yfinance returns a different structure
    if len(tickers) == 1:
        ticker = tickers[0]
        # For single ticker, all_data is a DataFrame without the outer 'ticker' level
        df_dict = {ticker: all_data}
    else:
        # For multiple tickers, all_data is a MultiIndex DataFrame: (ticker, price column)
        # Convert to dict of DataFrames per ticker
        df_dict = {}
        for ticker in tickers:
            try:
                # Extract the slice for this ticker
                ticker_data = all_data[ticker].copy()
                # Ensure it's a DataFrame (it should be)
                if not ticker_data.empty:
                    df_dict[ticker] = ticker_data
                else:
                    df_dict[ticker] = pd.DataFrame()
            except KeyError:
                # Ticker not found in the batch download
                df_dict[ticker] = pd.DataFrame()
                continue

    # ---------------------------
    # 2. Process each ticker using its downloaded DataFrame
    # ---------------------------
    for ticker in tickers:
        df_t = df_dict.get(ticker, pd.DataFrame())
        if df_t.empty:
            results.append({"Ticker": ticker, "Sector": TICKERS.get(ticker, "Unknown"), "Signal": "No data", "Prob": 0.0})
            continue

        # Ensure the index is datetime (already is from yfinance)
        # Add technical indicators
        df_t = safe_add_ta_features(df_t)

        # Basic macro for stock‑specific model
        basic_macro = macro_sector_df[['VIX', 'TNX', 'CL']]
        df_t_basic = df_t.copy()
        df_t_basic = df_t_basic.join(basic_macro, how='left').ffill().bfill()

        # ---------- Same logic as before (unchanged) ----------
        if alpha < 0.01:   # old per‑stock scanner
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
            results.append({
                "Ticker": ticker,
                "Sector": TICKERS.get(ticker, "Unknown"),
                "Signal": f"{prob:.1%}",
                "Prob": prob
            })

        else:   # pooled or hybrid model
            # Need fundamentals and sector ETF for enhanced features
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
                latest_enhanced = ensure_numeric(latest_enhanced)
                prob_raw = pooled_model.predict_proba(latest_enhanced)[0][1]

                if use_hybrid and 0 < alpha < 1:
                    stock_model, stock_feature_cols = get_stock_specific_model(ticker, df_t_basic)
                    latest_basic = df_t_basic[stock_feature_cols].fillna(0).iloc[[-1]]
                    latest_basic = ensure_numeric(latest_basic)
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
                latest = ensure_numeric(latest)
                prob_raw = ensemble_t.predict_proba(latest)[0][1]

            prob_cal = calibrate_prob(prob_raw, calibration_map)
            results.append({
                "Ticker": ticker,
                "Sector": TICKERS.get(ticker, "Unknown"),
                "Signal": f"{prob_raw:.1%}",
                "Prob": prob_raw,
                "Calibrated": prob_cal,
                "Position": f"{prob_cal:.0%}"
            })

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
                results = scan_tickers_fallback(ticker_list, macro_sector_df, alpha, multi_period, use_hybrid, profit_target)
                increment_user_scans(user_email)
                st.session_state.scanner_results = results
                st.rerun()
    else:
        with st.spinner(f"Scanning {len(ticker_list)} tickers... this may take a minute."):
            macro_sector_df = get_macro_sector_data_cached(multi_period)
            results = scan_tickers_fallback(ticker_list, macro_sector_df, alpha, multi_period, use_hybrid, profit_target)
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
            if pd.isna(val):
                return ''
            if isinstance(val, str):
                val = val.replace('%', '')
            pct = float(val) / 100
            if pct > 0.65:
                return 'background-color: #2e7d32; color: white; font-weight: bold;'
            elif pct < 0.35:
                return 'background-color: #c62828; color: white; font-weight: bold;'
            else:
                return 'background-color: #f5f5f5; color: #1e1e1e;'
        except:
            return ''

if not df_results.empty and 'Signal' in df_results.columns:
    styled_df = df_results.style.map(color_signal, subset=['Signal'])   # use .map, not .applymap
    st.dataframe(styled_df, width='stretch')
else:
    st.info("No scanner results to display.")

    st.dataframe(styled_df, width='stretch')

    if st.button("📥 Export Scanner Results to CSV"):
        df_results.to_csv("scanner_results.csv", index=False)
        st.success("✅ Saved as scanner_results.csv on your Desktop!")