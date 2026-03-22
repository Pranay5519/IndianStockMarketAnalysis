import streamlit as st
from utils.nlp import analyze_sentiment_with_langchain , fetch_news
from dotenv import load_dotenv
import os
load_dotenv()
st.set_page_config(page_title="GenAI Insights", page_icon="🤖")

st.title("🤖 Generative AI Market Sentiment")
st.write("Utilizing LangChain and Gemini 2.5 Flash to synthesize real-time news into actionable market sentiment.")

STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ITC": "ITC.NS",
    "Sun Pharma": "SUNPHARMA.NS"
}

selected_stock = st.selectbox("Select an Asset for AI Sentiment Analysis", list(STOCKS.keys()))
ticker = STOCKS[selected_stock]

st.divider()

# Secure API Key Handling for Reviewers
st.subheader("API Configuration")
st.info("To evaluate this module without local setup, please provide your Google API key below. Keys are not stored and are cleared upon exit.")

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    google_key = st.text_input("Google Gemini API Key:", type="password")
    st.caption("Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).")

if st.button("Generate AI Insight Report", use_container_width=True):
    if not google_key:
        st.warning("⚠️ A Google API key is required to run the Gemini model.")
    else:
        with st.spinner(f"Gemini is analyzing recent news for {selected_stock}..."):
            fetch_news()
            report = analyze_sentiment_with_langchain(ticker, selected_stock, google_key)
            
            if "Generative AI Error" in report or "Could not retrieve" in report:
                st.error(report)
            else:
                st.success("Analysis Complete!")
                st.markdown("### 📄 Gemini Sentiment Report")
                st.info(report)