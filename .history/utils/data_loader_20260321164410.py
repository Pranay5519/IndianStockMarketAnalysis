import yfinance as yf
import pandas as pd

STOCKS = {
    "HDFC BANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "SUN PHARMA": "SUNPHARMA.NS"
}

def load_stock_data(stock_name, period="1y", interval="1d"):
    
    ticker = STOCKS[stock_name]

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    df["Stock"] = stock_name

    return df