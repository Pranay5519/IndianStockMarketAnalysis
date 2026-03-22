from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import pandas as pd

def train_stacking_model(df):
    """Trains a Stacking Regressor to predict the next day's close price."""
    
    # Define features (X) and target (y)
    # Dropping Date, Stock, and the Target itself for training
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Return', 
                'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Trend']
    
    X = df[features]
    y = df['Target_Next_Close']
    
    # Time-series split (keeping chronological order)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Define Base Estimators
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
    ]
    
    # Define Meta Model
    meta_model = Ridge()
    
    # Build Stacking Regressor
    stacking_reg = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5
    )
    
    # Train the model
    stacking_reg.fit(X_train, y_train)
    
    # Evaluate
    predictions = stacking_reg.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return stacking_reg, mse, r2