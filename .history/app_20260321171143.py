# Add these to your existing imports
from utils.features import generate_technical_features
from models.stacking import train_stacking_model
import streamlit as st


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

# --- UI Layout: Trend & AI Predictions ---
st.subheader("🤖 AI/ML Predictive Analysis")

# Run the ML Model
with st.spinner('Training Stacking Regressor (LightGBM + Random Forest)...'):
    predicted_close, pct_change, model_r2 = get_ml_prediction(df, selected_stock_name)

# Create 4 columns for metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Trend", trend)
m2.metric("S/R Levels Found", f"{len(supports)} / {len(resistances)}")

# Display AI Prediction with dynamic delta color
m3.metric(label="AI Predicted Close (Next Day)", 
            value=f"₹{predicted_close:.2f}", 
            delta=f"{pct_change:.2f}% expected")

# Display Model Confidence
m4.metric("Model R² Score", f"{model_r2:.2f}")

st.info(f"**Technical Insight:** {insight}")
st.caption("Note: The AI prediction utilizes a Stacking Regressor combining LightGBM and Random Forest on trailing technical indicators. Do not use for actual trading.")