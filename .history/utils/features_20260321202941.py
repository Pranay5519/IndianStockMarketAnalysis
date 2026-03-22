import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import IsolationForest

def generate_technical_features(df, stock_symbol):
    """Calculates technical indicators and prepares the dataset for ML."""
    df = df.copy()
    
    # Add Stock identifier
    df['Stock'] = stock_symbol
    
    # Calculate Returns
    df['Return'] = df['Close'].pct_change()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Return'].rolling(window=20).std()
    
    # RSI & MACD
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd_diff() # Using the histogram/difference for ML
    
    # Trend Mapping (Numeric for ML: 1 for Bullish, -1 for Bearish)
    # Simple logic: If Close > MA20 -> 1, else -1
    df['Trend'] = np.where(df['Close'] > df['MA20'], 1, -1)
    
    # Target Variable: Next day's closing price
    df['Target_Next_Close'] = df['Close'].shift(-1)
    
    # Drop rows with NaN values created by rolling windows and shifting
    df.dropna(inplace=True)
    
    # Reorder columns to match your requirement
    cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Stock', 
            'Return', 'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Trend', 'Target_Next_Close']
    
    return df[cols]


def detect_anomalies(df):
    """Uses an Isolation Forest to detect anomalous trading days."""
    df = df.copy()
    
    # We need Return and Volatility features for the model to find anomalies
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()
    if 'Volatility' not in df.columns:
        df['Volatility'] = df['Return'].rolling(window=20).std()
        
    df.dropna(inplace=True)
    
    # Features to check for anomalies (Price swings and Volume spikes)
    features = ['Return', 'Volume', 'Volatility']
    X = df[features]
    
    # Initialize Isolation Forest (flagging the top 3% most extreme days)
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    
    # Fit and Predict (-1 means anomaly, 1 means normal)
    df['Anomaly'] = iso_forest.fit_predict(X)
    
    return df