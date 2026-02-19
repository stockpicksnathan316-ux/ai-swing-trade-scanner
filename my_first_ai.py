import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from ta import add_all_ta_features

st.set_page_config(page_title="AI Momentum Predictor")
st.title("🤖 AI Momentum Predictor")

ticker = st.text_input("Stock", "AAPL")
period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)

# --- LOAD DATA ---
df = yf.download(ticker, period=period, progress=False)

# --- FLATTEN MULTIINDEX ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# --- ADD TECHNICAL INDICATORS (20+ features) ---
df = add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume",
    fillna=True
)

# --- CREATE TARGET: price up 5 days later? ---
df['future_close'] = df['Close'].shift(-5)
df['target'] = (df['future_close'] > df['Close']).astype(int)

# --- DROP ROWS WITH NAN (from shifting and indicators) ---
# --- TRAIN/TEST SPLIT: NO CHEATING ---
# --- CHRONOLOGICAL TRAIN/TEST SPLIT (80% train, 20% test) ---
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

st.write(f"📅 Training: {df_train.index[0].date()} to {df_train.index[-1].date()} ({len(df_train)} days, 80%)")
st.write(f"📅 Testing:  {df_test.index[0].date()} to {df_test.index[-1].date()} ({len(df_test)} days, 20%)")

# Features & targets
feature_columns = [col for col in df.columns if col not in 
                   ['Open','High','Low','Close','Volume','future_close','target']]

X_train = df_train[feature_columns]
y_train = df_train['target']
X_test = df_test[feature_columns]
y_test = df_test['target']

# Train ONLY on historical data
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Predict on test data (unseen during training)
y_pred_proba = model.predict_proba(X_test)[:, 1]
df_test['prediction'] = y_pred_proba

# --- LIVE PREDICTION (TODAY) ---
latest_row = df[feature_columns].iloc[[-1]]  # Note the double brackets!
live_prob = model.predict_proba(latest_row)[0][1]

# --- BACKTEST ACCURACY ---
from sklearn.metrics import accuracy_score, precision_score
y_pred_class = (y_pred_proba > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_class)
prec = precision_score(y_test, y_pred_class, zero_division=0)

st.metric("🎯 Test Accuracy", f"{acc:.1%}")
st.metric("⚡ Test Precision", f"{prec:.1%}")
# Also show prediction as a gauge

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price'
))

# Optional: overlay traditional RSI signals (RSI<30)
if 'rsi' in df.columns:
    df['rsi'] = df['rsi']  # from ta library
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

# --- DISPLAY PREDICTIONS ---
col1, col2 = st.columns(2)
with col1:
    st.metric("🔮 Today's 5-Day UP Probability", f"{live_prob:.1%}")
with col2:
    st.metric("📊 Model trained on", f"{len(df_train)} days")

# --- SHOW FEATURE IMPORTANCE (optional) ---
if st.checkbox("Show what the AI learned"):
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    st.bar_chart(importance.set_index('feature'))