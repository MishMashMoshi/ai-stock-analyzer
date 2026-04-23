import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import numpy as np

st.title("📊 AI Stock Analyzer Dashboard (ULTIMATE FINAL)")
st.markdown("Quant System: Signals + Portfolio + Backtesting + Risk + AI Insights")

st.sidebar.subheader("🕒 Live Market Clock")
st.sidebar.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

auto_refresh = st.sidebar.toggle("Auto Refresh", value=False)
refresh_rate = st.sidebar.selectbox("Refresh Rate", [5,10,30,60], index=1)

stock_categories = {
    "Big Tech 🧠": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA"],
    "Growth 🚀": ["TSLA","AMD","PLTR","SNOW","CRWD"],
    "Safe 🏦": ["KO","PEP","JNJ","PG","WMT"],
    "ETFs 📊": ["SPY","QQQ","VOO","VTI"]
}

all_tickers = sorted(set(sum(stock_categories.values(), [])))

selected_category = st.sidebar.selectbox("Category", ["All"] + list(stock_categories.keys()))
tickers = all_tickers if selected_category == "All" else stock_categories[selected_category]

selected_ticker = st.selectbox("Stock", tickers)

data = {}

for ticker in tickers:
    df = yf.download(ticker, start="2022-01-01")

    if df.empty:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    # MAs
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    data[ticker] = df

df = data[selected_ticker].dropna()

df["Buy"] = (
    (df["RSI"] < 30) &
    (df["MACD"] > df["Signal"]) &
    (df["MA20"] > df["MA50"])
)

df["Returns"] = df["Close"].pct_change()

df["Strategy"] = df["Returns"] * df["Buy"].shift(1)

df["Cum_Strategy"] = (1 + df["Strategy"]).cumprod()
df["Cum_Market"] = (1 + df["Returns"]).cumprod()

rolling_max = df["Cum_Strategy"].cummax()
drawdown = (df["Cum_Strategy"] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

win_rate = (df["Strategy"] > 0).sum() / df["Strategy"].count()

latest_rsi = df["RSI"].iloc[-1]
latest_macd = df["MACD"].iloc[-1]
latest_signal = df["Signal"].iloc[-1]

def generate_explanation():
    explanation = []

    if latest_rsi < 30:
        explanation.append("Stock is oversold (RSI < 30)")
    elif latest_rsi > 70:
        explanation.append("Stock is overbought (RSI > 70)")

    if latest_macd > latest_signal:
        explanation.append("Momentum is bullish (MACD crossover)")
    else:
        explanation.append("Momentum is weakening")

    if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
        explanation.append("Uptrend confirmed (MA20 > MA50)")
    else:
        explanation.append("Downtrend detected")

    return " | ".join(explanation)

ai_explanation = generate_explanation()

trades = df[df["Buy"] == True][["Close", "RSI", "MACD"]].copy()
trades["Date"] = trades.index

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📉 Risk",
    "📜 Backtest",
    "🧠 AI Insight",
    "📋 Trades"
])

with tab1:
    st.subheader("Price")
    st.line_chart(df["Close"])

with tab2:

    col1, col2 = st.columns(2)

    col1.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
    col2.metric("Win Rate", f"{win_rate*100:.2f}%")

with tab3:

    col1, col2 = st.columns(2)

    col1.metric("Strategy Return", f"{(df['Cum_Strategy'].iloc[-1]-1)*100:.2f}%")
    col2.metric("Market Return", f"{(df['Cum_Market'].iloc[-1]-1)*100:.2f}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["Cum_Strategy"], name="Strategy"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Cum_Market"], name="Market"))

    st.plotly_chart(fig, use_container_width=True)

with tab4:

    st.subheader("🧠 AI Explanation")
    st.write(ai_explanation)

with tab5:

    st.subheader("📋 Trade Signals")
    st.dataframe(trades.tail(20))

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
