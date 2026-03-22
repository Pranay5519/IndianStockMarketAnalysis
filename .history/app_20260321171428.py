import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# Import our custom ML modules
from utils.features import generate_technical_features
from models.stacking import train_stacking_model

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
    
    # Flatten MultiIndex columns returned by newer versions of yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    return df

# --- Technical Analysis ---
def calculate_trend_and_indicators(df):
    """Calculates SMA and RSI to determine the current trend."""
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
    
    resistances = df[df['High'] == highs]['High'].dropna().unique()
    supports = df[df['Low'] == lows]['Low'].dropna().unique()
    
    top_resistances = sorted(resistances)[-2:] if len(resistances) >= 2 else resistances
    bottom_supports = sorted(supports)[:2] if len(supports) >= 2 else supports
    
    return bottom_supports, top_resistances

# --- Machine Learning Integration ---
@st.cache_resource(ttl=3600)
def get_ml_prediction(stock_symbol):
    """Fetches 2Y historical data, generates features, trains the stack, and predicts."""
    # 1. Fetch a dedicated long-term dataset for ML training
    ticker = STOCKS[stock_symbol]
    ml_df = fetch_data(ticker, period="2y") 
    
    # 2. Generate Features
    processed_df = generate_technical_features(ml_df, stock_symbol)
    
    # Check if we have enough data to prevent crashes
    if len(processed_df) < 10:
        return 0.0, 0.0, 0.0
        
    # 3. Train Model
    model, mse, r2 = train_stacking_model(processed_df)
    
    # 4. Predict the next day
    latest_features = processed_df.iloc[-1:][['Close', 'High', 'Low', 'Open', 'Volume', 'Return', 
                                              'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Trend']]
    
    predicted_close = model.predict(latest_features)[0]
    current_close = processed_df.iloc[-1]['Close']
    pct_change = ((predicted_close - current_close) / current_close) * 100
    
    return predicted_close, pct_change, r2

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

# --- Main App Execution ---
st.title("📈 Indian Stock Market Analysis Module")
st.write("Analyze NSE-listed equities with automated S/R detection and AI-driven predictions.")

col1, col2 = st.columns([1, 1])
selected_stock_name = col1.selectbox("Select Stock", list(STOCKS.keys()))
selected_timeframe_name = col2.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))

ticker = STOCKS[selected_stock_name]
period = TIMEFRAMES[selected_timeframe_name]

with st.spinner('Fetching market data...'):
    df = fetch_data(ticker, period)

if not df.empty:
    # Processing Core Logic
    df, trend, insight = calculate_trend_and_indicators(df)
    supports, resistances = calculate_support_resistance(df)

    # UI Layout: Trend & AI Predictions
    st.subheader("🤖 AI/ML Predictive Analysis")
    
    with st.spinner('Training Stacking Regressor (LightGBM + Random Forest)...'):
        predicted_close, pct_change, model_r2 = get_ml_prediction(selected_stock_name)    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Trend", trend)
    m2.metric("S/R Levels Found", f"{len(supports)} / {len(resistances)}")
    m3.metric(label="AI Predicted Close (Next Day)", 
              value=f"₹{predicted_close:.2f}", 
              delta=f"{pct_change:.2f}% expected")
    m4.metric("Model R² Score", f"{model_r2:.2f}")
    
    st.info(f"**Technical Insight:** {insight}")
    st.caption("Note: The AI prediction utilizes a Stacking Regressor combining LightGBM and Random Forest on trailing technical indicators. Do not use for actual trading.")

    # UI Layout: Chart
    st.plotly_chart(plot_interactive_chart(df, selected_stock_name, supports, resistances), use_container_width=True)

else:
    st.error("Failed to retrieve data. Please try again.")