import yfinance as yf
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # Or substitute with your preferred LLM
import os

def get_stock_news(ticker):
    """Fetches recent news headlines for the stock using yfinance."""
    stock = yf.Ticker(ticker)
    news = stock.news
    if not news:
        return "No recent news found."
    
    headlines = [item['title'] for item in news[:5]]
    return "\n".join(f"- {h}" for h in headlines)

def analyze_sentiment_with_langchain(ticker, stock_name, api_key):
    """Uses LangChain to summarize news and determine sentiment."""
    # 1. Fetch News
    news_text = get_stock_news(ticker)
    
    if news_text == "No recent news found.":
        return "Not enough current data to generate an AI sentiment report."

    # 2. Initialize LLM
    llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-2.5-flash", api=api_key)

    # 3. Create LangChain Prompt
    template = """
    You are an expert financial AI assistant. Read the following recent news headlines for {stock_name} ({ticker}).
    
    Recent News:
    {news}
    
    Task:
    1. Provide a 2-3 sentence summary of the current market narrative around this stock.
    2. Classify the overall sentiment as BULLISH, BEARISH, or NEUTRAL.
    3. Identify any potential risks or catalysts mentioned in the headlines.
    
    Format the output cleanly using Markdown.
    """
    
    prompt = PromptTemplate(
        input_variables=["stock_name", "ticker", "news"],
        template=template
    )
    
    # 4. Execute Chain
    chain = prompt | llm
    response = chain.invoke({
        "stock_name": stock_name, 
        "ticker": ticker, 
        "news": news_text
    })
    
    return response.content