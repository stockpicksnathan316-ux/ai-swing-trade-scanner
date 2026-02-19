import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.title("🔧 AI Chart - Debug Mode")

ticker = st.text_input("Stock", "AAPL")
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)

# Load data with error catching
try:
    df = yf.download(ticker, period=period, progress=False)
    st.write(f"✅ Data downloaded! Shape: {df.shape}")
    st.write("First 3 rows:", df.head(3))
except Exception as e:
    st.error(f"❌ Failed to download data: {e}")
    st.stop()

# Check if data is empty
if df.empty:
    st.error("❌ DataFrame is empty. Yahoo Finance returned no data.")
    st.stop()

# Calculate RSI safely
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Signal: RSI < 30
df['Signal'] = (df['RSI'] < 30).astype(int)

# Show signal count
st.write(f"📊 Buy signals detected: {df['Signal'].sum()}")

# Create plot
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price'
))

# Only plot signals if there are any
if df['Signal'].sum() > 0:
    fig.add_trace(go.Scatter(
        x=df.index[df['Signal'] == 1],
        y=df['Close'][df['Signal'] == 1] * 0.98,  # slightly below price for visibility
        mode='markers',
        marker=dict(size=12, color='lime', symbol='triangle-up'),
        name='Buy Signal'
    ))
else:
    st.warning("⚠️ No buy signals found in this period.")

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Optional: raw data table
if st.checkbox("Show raw data"):
    st.dataframe(df.tail(10))