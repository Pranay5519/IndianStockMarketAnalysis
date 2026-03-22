from xgboost import XGBRegressor

def train_model(df):

    df = df.dropna().copy()

    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()

    features = [
        "Open","High","Low","Close","Volume",
        "MA20","MA50","RSI","MACD","Volatility"
    ]

    X = df[features]
    y = df["Target"]

    split = int(len(df)*0.8)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5
    )

    model.fit(X[:split], y[:split])

    return model