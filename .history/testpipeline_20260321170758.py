import yfinance as yf
from utils.features import generate_technical_features
from models.stacking import train_stacking_model

def run_test():
    print("1. Fetching Data...")
    ticker = "RELIANCE.NS"
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    
    # Handle multi-index columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    
    print("2. Generating Features...")
    processed_df = generate_technical_features(df, ticker)
    
    print("--- Processed Data Sample ---")
    print(processed_df.head(3))
    print(f"Dataset shape after dropping NaNs: {processed_df.shape}\n")
    
    print("3. Training Stacking Regressor...")
    model, mse, r2 = train_stacking_model(processed_df)
    
    print("--- Model Evaluation ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    if r2 > 0:
        print("✅ Pipeline executed successfully! Model is learning.")
    else:
        print("⚠️ Pipeline ran, but model R2 is poor. Might need more data or feature tuning.")

if __name__ == "__main__":
    run_test()