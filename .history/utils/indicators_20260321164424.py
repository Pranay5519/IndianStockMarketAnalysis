import ta
import pandas as pd

def add_indicators(df):

    df = df.sort_values("Date")

    df["Return"] = df["Close"].pct_change()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Volatility"] = df["Return"].rolling(20).std()

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()

    return df