import streamlit as st

st.set_page_config(page_title="FinVise AI | Home", page_icon="🏠", layout="wide")

st.title("FinVise AI: Indian Stock Market Analysis")
st.markdown("### Full Stack AI/ML Technical Assignment")

st.write("""
Welcome to the FinVise AI Market Analysis Module. Use the sidebar to navigate between:
* **📈 Analysis:** Core interactive charting, automated Support/Resistance detection, and a Stacking Regressor (LightGBM + Random Forest) for predictive price modeling.
* **🤖 GenAI Insights:** An NLP pipeline utilizing LangChain to parse real-time market news and extract human-readable sentiment.
""")

st.divider()

st.subheader("Selected Stock Universe Justification")
st.write("To demonstrate the versatility of the solution across various market dynamics[cite: 11], the following 5 NSE-listed equities were selected:")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    * **Reliance Industries (Energy/Conglomerate):** High market cap, high liquidity, sensitive to global commodity cycles.
    * **TCS (IT Services):** Export-driven, sensitive to global macroeconomic shifts and USD/INR rates.
    * **HDFC Bank (Financials):** Heavyweight banking stock, excellent for testing volatility and interest rate sensitivity.
    """)
with col2:
    st.markdown("""
    * **ITC (FMCG):** Defensive stock known for steady dividends and resistance to broader market corrections.
    * **Sun Pharma (Healthcare):** Introduces regulatory and US FDA compliance news catalysts into the NLP model.
    """)