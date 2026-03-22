# Add these to your existing imports
from utils.features import generate_technical_features
from models.stacking import train_stacking_model
import stre

@st.cache_resource(ttl=3600) # Cache the model so it doesn't retrain on every UI click
def get_ml_prediction(df, stock_symbol):
    """Generates features, trains the stack, and predicts tomorrow's close."""
    # 1. Generate Features
    processed_df = generate_technical_features(df, stock_symbol)
    
    # 2. Train Model
    model, mse, r2 = train_stacking_model(processed_df)
    
    # 3. Predict the next day
    # Get the very last row of our features to predict tomorrow
    latest_features = processed_df.iloc[-1:][['Close', 'High', 'Low', 'Open', 'Volume', 'Return', 
                                              'MA20', 'MA50', 'Volatility', 'RSI', 'MACD', 'Trend']]
    
    predicted_close = model.predict(latest_features)[0]
    current_close = processed_df.iloc[-1]['Close']
    
    # Calculate expected percentage move
    pct_change = ((predicted_close - current_close) / current_close) * 100
    
    return predicted_close, pct_change, r2