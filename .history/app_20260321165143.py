import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_stock_data, STOCKS
from utils.indicators import add_indicators
from utils.support_resistance import get_support_resistance
from utils.prediction import train_model
from utils.sentiment import fetch_news, sentiment_score, COMPANY_NAMES

st.set_page_config(layout="wide")

st.title("📈 Indian Stock AI Analysis Dashboard")

# ⭐ Sidebar
stock_name = st.sidebar.selectbox(
    "Select Stock",
    list(STOCKS.keys())
)

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["6mo", "1y"]
)

# ⭐ Load Data
df = load_stock_data(stock_name, period=timeframe)

df = add_indicators(df)

df = df.dropna()

# ⭐ Support Resistance
support, resistance = get_support_resistance(df)

# ⭐ Train Model
model = train_model(df)

features = [
    "Open","High","Low","Close","Volume",
    "MA20","MA50","RSI","MACD","Volatility"
]

latest = df[features].iloc[-1:]

prediction = model.predict(latest)[0]

# ⭐ News Sentiment
company = COMPANY_NAMES[stock_name]

headlines = fetch_news(company)

#score = sentiment_score(headlines)

# ⭐ Chart
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
))

# MA lines
fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MA20"],
    name="MA20"
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MA50"],
    name="MA50"
))

# Support lines
for s in support:
    fig.add_hline(y=s, line_dash="dash", line_color="green")

# Resistance lines
for r in resistance:
    fig.add_hline(y=r, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)

# ⭐ Insights Section
col1, col2 = st.columns(2)

col1.metric("Current Price", round(df["Close"].iloc[-1],2))
col2.metric("Predicted Next Close", round(prediction,2))
col3.metric("News Sentiment Score", round(score,2))

# ⭐ Headlines
st.subheader("Latest News Headlines")

for h in headlines[:5]:
    st.write("•", h)