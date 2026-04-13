# Day 8 Progress:
# - Built interactive Streamlit dashboard
# - Integrated stock selection (AAPL, MSFT, GOOGL)
# - Calculated RSI and MACD manually (no ta dependency)
# - Fixed yfinance multi-index issue
# - Displayed price, RSI, and MACD charts
# - Created real-time stock metrics (return, volatility)
# - Debugged full data pipeline (major milestone)
# Day 9 Progress:
# - Added AI-based BUY / SELL / HOLD signal system
# - Integrated RSI and MACD into decision logic
# - Transformed dashboard from visualization → decision-making tool
# - Improved user experience with clear stock recommendations
# Day 10 Progress:
# - Built multi-stock comparison system
# - Added scoring model combining return, volatility, and RSI
# - Implemented automatic “Best Stock Today” selector
# - Upgraded app from single-stock analysis to portfolio intelligence tool
# Day 11 Progress:
# - Added confidence scoring system (0–100 scale)
# - Combined RSI + MACD into weighted decision model
# - Improved AI recommendation transparency
# - Enhanced dashboard readability and user understanding
# Day 12 Progress:
# - Improved Streamlit UI layout with columns and metrics
# - Added color-coded BUY/SELL/HOLD status indicators
# - Introduced dashboard-style structure for readability
# - Enhanced user experience with cleaner visual hierarchy
# Day 13 Progress:
# - Added featured “Top Stock Today” highlight card
# - Improved ranking visibility in UI
# - Enhanced readability of AI-generated stock selection
# - Began transforming dashboard into a structured product interface
# Day 14 Progress:
# - Built stock comparison table across multiple tickers
# - Combined return, volatility, RSI, and scoring into unified dataset
# - Added DataFrame-based analytics view to Streamlit dashboard
# - Improved multi-stock analysis UX and readability
# Day 15 Progress:
# - Built AI explanation engine for stock selection
# - Converted raw metrics into human-readable insights
# - Added reasoning layer to best stock output
# - Improved interpretability of financial signals
# Day 16 Progress:
# - Redesigned Streamlit layout for dashboard-style UI
# - Added column-based structure for metrics and signals
# - Improved visual hierarchy with section dividers
# - Enhanced “Top Stock” presentation for clarity and impact

import streamlit as st
import yfinance as yf
import pandas as pd

# ---------------- APP HEADER ----------------
st.title("📊 AI Stock Analyzer Dashboard")
st.markdown("AI-powered stock analysis using RSI, MACD, volatility, and momentum scoring")

tickers = ["AAPL", "MSFT", "GOOGL"]
selected_ticker = st.selectbox("Choose a stock", tickers)

# ---------------- DATA CACHE ----------------
data = {}

for ticker in tickers:
    df_temp = yf.download(ticker, start="2023-01-01", end="2026-04-01")

    # flatten columns if needed
    if isinstance(df_temp.columns, pd.MultiIndex):
        df_temp.columns = df_temp.columns.get_level_values(0)

    close = df_temp["Close"].squeeze().astype(float)

    # ---------------- RSI ----------------
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df_temp["RSI"] = 100 - (100 / (1 + rs))

    # ---------------- MACD ----------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df_temp["MACD"] = ema12 - ema26
    df_temp["MACD_Signal"] = df_temp["MACD"].ewm(span=9, adjust=False).mean()

    # store clean
    data[ticker] = {
        "df": df_temp,
        "close": close
    }

# ---------------- SELECTED STOCK ----------------
df = data[selected_ticker]["df"]
close = data[selected_ticker]["close"]

selected_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
selected_volatility = close.pct_change().std()

selected_rsi = df["RSI"].iloc[-1]
selected_macd = df["MACD"].iloc[-1]
selected_signal = df["MACD_Signal"].iloc[-1]

# ---------------- SIGNAL ----------------
signal = "HOLD"

if selected_rsi < 30 and selected_macd > selected_signal:
    signal = "BUY 🟢"
elif selected_rsi > 70 and selected_macd < selected_signal:
    signal = "SELL 🔴"

# ---------------- UI METRICS ----------------
st.subheader(f"{selected_ticker} Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Return", f"{selected_change:.2f}%")
col2.metric("Volatility", f"{selected_volatility:.4f}")
col3.metric("RSI", f"{selected_rsi:.2f}")

st.subheader("📢 AI Recommendation")

if signal == "BUY 🟢":
    st.success(signal)
elif signal == "SELL 🔴":
    st.error(signal)
else:
    st.warning(signal)

# ---------------- CONFIDENCE ----------------
confidence = 50

if selected_rsi < 30:
    confidence += 25
elif selected_rsi > 70:
    confidence -= 25

if selected_macd > selected_signal:
    confidence += 15
else:
    confidence -= 15

confidence = max(0, min(100, confidence))

st.write(f"📊 Confidence Score: {confidence}/100")

# ---------------- COMPARISON ENGINE ----------------
comparison_data = []

for ticker in tickers:
    df_temp = data[ticker]["df"]
    close = data[ticker]["close"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    volatility = close.pct_change().std()

    score = change - (volatility * 100)

    if rsi < 30:
        score += 5
    elif rsi > 70:
        score -= 5

    comparison_data.append([
        ticker,
        round(change, 2),
        round(volatility, 4),
        round(rsi, 2),
        round(score, 2)
    ])

# ---------------- TABLE ----------------
comparison_df = pd.DataFrame(
    comparison_data,
    columns=["Stock", "Return %", "Volatility", "RSI", "Score"]
)

comparison_df = comparison_df.sort_values(by="Score", ascending=False)

st.subheader("📊 Stock Comparison Leaderboard")
st.dataframe(comparison_df)

# ---------------- BEST STOCK ----------------
best_stock = comparison_df.iloc[0]["Stock"]
st.markdown("## 🏆 Top Stock Today")
st.success(f"{best_stock} is currently the strongest stock")

# ---------------- EXPLANATION ----------------
st.subheader("🧠 AI Explanation")

best_close = data[best_stock]["close"]

best_delta = best_close.diff()
best_gain = best_delta.clip(lower=0)
best_loss = -best_delta.clip(upper=0)

best_rs = best_gain.rolling(14).mean() / best_loss.rolling(14).mean()
best_rsi = (100 - (100 / (1 + best_rs))).iloc[-1]

reasons = []

if best_rsi < 30:
    reasons.append("oversold (potential rebound)")
elif best_rsi > 70:
    reasons.append("overbought (possible pullback)")
else:
    reasons.append("healthy RSI range")

best_change = ((best_close.iloc[-1] - best_close.iloc[0]) / best_close.iloc[0]) * 100
best_vol = best_close.pct_change().std()

if best_change > 0:
    reasons.append("positive momentum")
else:
    reasons.append("negative trend")

if best_vol < 0.02:
    reasons.append("low volatility (stable)")
else:
    reasons.append("higher volatility (riskier)")

for r in reasons:
    st.write("• " + r)

# ---------------- CHARTS ----------------
st.subheader("Price Chart")
st.line_chart(close)

st.subheader("RSI Chart")
st.line_chart(df["RSI"])

st.subheader("MACD Chart")
st.line_chart(df[["MACD", "MACD_Signal"]])