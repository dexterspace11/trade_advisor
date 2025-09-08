import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="RSI-Adjusted S/R Advisor", layout="wide")
st.title("RSI-Adjusted Support/Resistance Trading Advisor")

# =========================
# ===== Asset Directory ===
# =========================
ASSET_DIRECTORY = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "XRP (XRP-USD)": "XRP-USD",
    "Euro/USD (EURUSD=X)": "EURUSD=X",
    "Gold (GC=F)": "GC=F",
    "Crude Oil (CL=F)": "CL=F",
    "S&P 500 (^GSPC)": "^GSPC"
}

# =========================
# ===== User Inputs =======
# =========================
colI1, colI2, colI3 = st.columns([2,1,1])
with colI1:
    asset_choice = st.selectbox("Choose Asset (Auto fills ticker)", list(ASSET_DIRECTORY.keys()))
    ticker = st.text_input("Or enter custom Yahoo Finance Ticker", value=ASSET_DIRECTORY[asset_choice])
with colI2:
    period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=0)  # default 1mo
with colI3:
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=1)  # default 5m

autofix = st.checkbox("Auto-fix invalid period/interval combos", value=True)

colP1, colP2, colP3 = st.columns(3)
with colP1:
    rsiPeriod = st.number_input("RSI Period", min_value=2, max_value=100, value=14)
with colP2:
    lookback = st.number_input("Lookback for Support/Resistance", min_value=5, max_value=1000, value=50)
with colP3:
    smoothLength = st.number_input("Smoothing Length", min_value=1, max_value=200, value=5)

# =========================
# ===== Data Loader =======
# =========================
@st.cache_data(ttl=600, show_spinner=True)
def load_data(sym: str, per: str, itv: str, fix: bool = True) -> pd.DataFrame:
    attempts = []
    tried = []

    combos = [(per, itv)]
    if fix:
        combos.extend([
            ("1y", "1d"),
            ("6mo", "1d"),
            ("1mo", "1h"),
            ("7d", "1h"),
            ("60d", "5m"),
        ])

    for p, i in combos:
        try:
            df = yf.download(sym, period=p, interval=i, auto_adjust=True, progress=False)
            tried.append((p, i, len(df)))
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, tried
        except Exception as e:
            attempts.append(f"{p}/{i} failed: {e}")

    return pd.DataFrame(), tried

prices, tried = load_data(ticker, period, interval, autofix)

st.write("### Data Fetch Debug")
for p, i, rows in tried:
    st.text(f"Tried Period={p}, Interval={i} → Rows={rows}")

if prices.empty:
    st.error(
        "Unable to fetch data with the tried period/interval combinations. "
        "Please try again with a different ticker, shorter interval, or longer period."
    )
    st.stop()

# =========================
# ===== Indicators =========
# =========================
def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=close.index).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=close.index).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

prices = prices.rename(columns={"Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"})
prices["RSI"] = rsi_series(prices["Close"], rsiPeriod)
prices["RSI_diff"] = (prices["RSI"] - 50).abs() / 50

prices["Base_Support"] = prices["Close"].rolling(lookback, min_periods=lookback).min()
prices["Base_Resistance"] = prices["Close"].rolling(lookback, min_periods=lookback).max()
prices["Range"] = prices["Base_Resistance"] - prices["Base_Support"]

prices["Adj_Support"] = prices["Base_Support"] - (prices["RSI_diff"] * prices["Range"])
prices["Adj_Resistance"] = prices["Base_Resistance"] + (prices["RSI_diff"] * prices["Range"])
prices["Midline"] = (prices["Adj_Support"] + prices["Adj_Resistance"]) / 2

prices["Smooth_Support"] = prices["Adj_Support"].rolling(smoothLength, min_periods=smoothLength).mean()
prices["Smooth_Resistance"] = prices["Adj_Resistance"].rolling(smoothLength, min_periods=smoothLength).mean()
prices["Smooth_Midline"] = prices["Midline"].rolling(smoothLength, min_periods=smoothLength).mean()

valid = prices.dropna(subset=["Smooth_Support", "Smooth_Resistance", "Smooth_Midline", "RSI"]).copy()

if valid.empty:
    st.warning(
        "Not enough bars after applying lookback/smoothing. Increase the period, reduce lookback/smoothing, or choose a higher timeframe."
    )
    st.stop()

valid["Buy_Signal"] = (
    (valid["Close"] > valid["Smooth_Support"]) &
    (valid["Close"].shift(1) <= valid["Smooth_Support"].shift(1)) &
    (valid["RSI"] > valid["RSI"].shift(1))
)
valid["Sell_Signal"] = (
    (valid["Close"] < valid["Smooth_Resistance"]) &
    (valid["Close"].shift(1) >= valid["Smooth_Resistance"].shift(1)) &
    (valid["RSI"] < valid["RSI"].shift(1))
)

latest = valid.iloc[-1]

# =========================
# ===== Recommendations ===
# =========================
st.subheader("📊 Trading Advisor Recommendation")
colR1, colR2, colR3 = st.columns(3)

if latest["Buy_Signal"]:
    colR1.success(f"Action: BUY @ ~{latest['Close']:.4f}")
elif latest["Sell_Signal"]:
    colR1.error(f"Action: SELL @ ~{latest['Close']:.4f}")
else:
    colR1.info("Action: WAIT — No fresh signal on the latest bar.")

buy_level = latest["Smooth_Support"]
sell_level = latest["Smooth_Resistance"]
colR2.metric("Suggested Buy Level (Smoothed Support)", f"{buy_level:.4f}")
colR3.metric("Suggested Sell Level (Smoothed Resistance)", f"{sell_level:.4f}")

colS1, colS2, colS3, colS4 = st.columns(4)
colS1.metric("Close", f"{latest['Close']:.4f}")
colS2.metric("RSI(14)", f"{latest['RSI']:.2f}")
colS3.metric("Midline", f"{latest['Smooth_Midline']:.4f}")
colS4.metric("Range (Base)", f"{latest['Range']:.4f}")

with st.expander("More Levels (base vs adjusted)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"Base Support: **{latest['Base_Support']:.4f}**")
    c2.write(f"Base Resistance: **{latest['Base_Resistance']:.4f}**")
    c3.write(f"Adjusted Support: **{latest['Adj_Support']:.4f}**")
    c4.write(f"Adjusted Resistance: **{latest['Adj_Resistance']:.4f}**")

# =========================
# ===== Chart =========
# =========================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(valid.index, valid['Close'], label='Close')
ax.plot(valid.index, valid['Smooth_Support'], label='Smoothed Support')
ax.plot(valid.index, valid['Smooth_Resistance'], label='Smoothed Resistance')
ax.plot(valid.index, valid['Smooth_Midline'], label='Smoothed Midline')

ax.plot(valid[valid['Buy_Signal']].index, valid[valid['Buy_Signal']]['Close'], '^', markersize=9, label='Buy Signal')
ax.plot(valid[valid['Sell_Signal']].index, valid[valid['Sell_Signal']]['Close'], 'v', markersize=9, label='Sell Signal')

ax.set_title(f"{ticker} — RSI-Adjusted Support/Resistance")
ax.set_ylabel("Price")
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# =========================
# ===== Recent Table ======
# =========================
st.subheader("Recent Signals & Levels")
st.dataframe(valid.tail(20)[[
    'Close','RSI','Smooth_Support','Smooth_Resistance','Smooth_Midline','Buy_Signal','Sell_Signal'
]])

# =========================
# ===== Tips Section ======
# =========================
st.markdown("## 💡 Tips")
st.markdown("""
- For **aggressive investing/trading**, set **Interval = 5 minutes**.  
- For **conservative investing**, set **Interval = 1 hour or more**.  
""")

st.markdown("## ⚠️ Important Tips")
st.markdown("""
- If **Close < Midline**, the price is likely to reach the **suggested Buy level (Support)**.  
- If **Close > Midline**, the price is likely to reach the **suggested Sell level (Resistance)**.  
""")

