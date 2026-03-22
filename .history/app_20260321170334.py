import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# --- Configuration & State ---
st.set_page_config(page_title="Indian Stock Market Analysis", layout="wide")

STOCKS = {
    "Reliance Industries (Energy)": "RELIANCE.NS",
    "TCS (IT)": "TCS.NS",
    "HDFC Bank (Banking)": "HDFCBANK.NS",
    "ITC (FMCG)": "ITC.NS",
    "Sun Pharma (Healthcare)": "SUNPHARMA.NS"
}

TIMEFRAMES = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_data(ticker, period):
    """Fetches historical OHLCV data from yfinance."""
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    
    # FIX: Flatten MultiIndex columns returned by newer versions of yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    return df

# --- Technical Analysis ---
def calculate_trend_and_indicators(df):
    """Calculates SMA and RSI to determine the current trend."""
    # Ensure we have enough data
    if len(df) < 50:
        return df, "Insufficient Data", "N/A"

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    current_close = df['Close'].iloc[-1].item()
    sma_20 = df['SMA_20'].iloc[-1].item()
    sma_50 = df['SMA_50'].iloc[-1].item()
    current_rsi = df['RSI'].iloc[-1].item()

    # Trend Logic
    if current_close > sma_20 and sma_20 > sma_50 and current_rsi < 70:
        trend = "🟢 Bullish"
    elif current_close < sma_20 and sma_20 < sma_50 and current_rsi > 30:
        trend = "🔴 Bearish"
    else:
        trend = "🟡 Sideways / Consolidating"

    insight = f"Current Price is ₹{current_close:.2f}. RSI is at {current_rsi:.2f}."
    return df, trend, insight

def calculate_support_resistance(df, window=20):
    """Finds local minima and maxima for Support/Resistance levels."""
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()
    
    # Extract unique levels
    resistances = df[df['High'] == highs]['High'].dropna().unique()
    supports = df[df['Low'] == lows]['Low'].dropna().unique()
    
    # Get the top 2 highest resistances and bottom 2 lowest supports
    top_resistances = sorted(resistances)[-2:] if len(resistances) >= 2 else resistances
    bottom_supports = sorted(supports)[:2] if len(supports) >= 2 else supports
    
    return bottom_supports, top_resistances

# --- Visualization ---
def plot_interactive_chart(df, ticker_name, supports, resistances):
    """Creates a Plotly chart with candlesticks, volume, and S/R levels."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker_name} Price', 'Volume'),
                        row_width=[0.2, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'].squeeze(), 
                                 high=df['High'].squeeze(), low=df['Low'].squeeze(), 
                                 close=df['Close'].squeeze(), name='Price'), row=1, col=1)

    # Volume
    colors = ['green' if row['Close'].item() >= row['Open'].item() else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'].squeeze(), marker_color=colors, name='Volume'), row=2, col=1)

    # Support / Resistance Lines
    for s in supports:
        fig.add_hline(y=s, line_dash="dash", line_color="green", annotation_text=f"Support: {s:.2f}", row=1, col=1)
    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="red", annotation_text=f"Resistance: {r:.2f}", row=1, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# --- Main App ---
st.title("📈 Indian Stock Market Analysis Module")
st.write("Analyze NSE-listed equities with automated S/R detection and trend insights.")

col1, col2 = st.columns([1, 1])
selected_stock_name = col1.selectbox("Select Stock", list(STOCKS.keys()))
selected_timeframe_name = col2.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))

ticker = STOCKS[selected_stock_name]
period = TIMEFRAMES[selected_timeframe_name]

with st.spinner('Fetching market data...'):
    df = fetch_data(ticker, period)

if not df.empty:
    # Processing
    df, trend, insight = calculate_trend_and_indicators(df)
    supports, resistances = calculate_support_resistance(df)

    # UI Layout: Trend Insights
    st.subheader("Market Sentiment & Trend Analysis")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Current Trend", trend)
    metric_col2.metric("Support Levels Found", len(supports))
    metric_col3.metric("Resistance Levels Found", len(resistances))
    st.info(f"**Automated Insight:** {insight}")

    # UI Layout: Chart
    st.plotly_chart(plot_interactive_chart(df, selected_stock_name, supports, resistances), use_container_width=True)

else:
    st.error("Failed to retrieve data. Please try again.")