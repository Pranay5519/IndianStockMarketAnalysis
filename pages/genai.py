import streamlit as st
from utils.nlp import analyze_sentiment_with_langchain, fetch_news

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

# 1. Fetch News API Key directly from Streamlit Secrets
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

st.divider()

# Secure API Key Handling for Reviewers
st.subheader("API Configuration")

# 2. Try to get Google Key from secrets, fallback to UI input if not found
google_key = st.secrets.get("GOOGLE_API_KEY", "")
if not google_key:
    google_key = st.text_input("Google Gemini API Key:", type="password")
    st.caption("Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).")

if st.button("Generate AI Insight Report", width="stretch"):
    if not google_key:
        st.warning("⚠️ A Google API key is required to run the Gemini model.")
    elif not NEWS_API_KEY:
        st.warning("⚠️ NEWS_API_KEY is missing from Streamlit secrets.")
    else:
        with st.spinner(f"Gemini is analyzing recent news for {selected_stock}..."):
                    # 1. Fetch the news
                    raw_news = fetch_news(selected_stock, NEWS_API_KEY)
                    
                    # 2. Format the news cleanly for both the UI and the LLM
                    if isinstance(raw_news, list) and len(raw_news) > 0:
                        # If fetch_news returns a list, format it with markdown bullet points
                        formatted_news = "\n".join([f"- {headline}" for headline in raw_news])
                    elif isinstance(raw_news, str):
                        # If fetch_news already returns a formatted string
                        formatted_news = raw_news
                    else:
                        formatted_news = "No recent news found."
                    
                    # 3. Generate the AI Report
                    report = analyze_sentiment_with_langchain(formatted_news, ticker, selected_stock, google_key)
                    
                    # 4. Display the results beautifully
                    if "Generative AI Error" in report or "Could not retrieve" in report:
                        st.error(report)
                    else:
                        st.success("Analysis Complete!")
                        
                        # Show the AI Report prominently
                        st.markdown("### 📄 Gemini Sentiment Report")
                        st.info(report)
                        
                        # Hide the raw headlines in a collapsible expander for a cleaner UI
                        with st.expander("📰 View Raw Headlines Analyzed"):
                            st.markdown(formatted_news)