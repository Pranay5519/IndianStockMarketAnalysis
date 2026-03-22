import streamlit as st
from utils.nlp import analyze_sentiment_with_langchain
import os
st.title("🤖 Generative AI Market Sentiment")
st.write("Utilizing LangChain to synthesize real-time news into actionable market sentiment.")

STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ITC": "ITC.NS",
    "Sun Pharma": "SUNPHARMA.NS"
}

selected_stock = st.selectbox("Select an Asset for AI Sentiment Analysis", list(STOCKS.keys()))
ticker = STOCKS[selected_stock]

# Secure API Key Handling
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your OpenAI API Key to run the LangChain pipeline:", type="password")
    st.caption("Your key is not stored and is only used for this session.")

if st.button("Generate AI Insight Report"):
    if not api_key:
        st.warning("An API key is required to connect to the Generative AI model.")
    else:
        with st.spinner(f"Agents are scouring recent news for {selected_stock}..."):
            try:
                report = analyze_sentiment_with_langchain(ticker, selected_stock, api_key)
                st.markdown("### 📄 LangChain Sentiment Report")
                st.info(report)
            except Exception as e:
                st.error(f"LangChain Pipeline Error: {e}")