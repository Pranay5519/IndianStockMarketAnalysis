import os
from newsapi import NewsApiClient
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def fetch_news(company, api_key):
    """Fetches recent news headlines for the company using NewsAPI."""
    try:
        newsapi = NewsApiClient(api_key=api_key)
        response = newsapi.get_everything(
            q=company,
            language="en",
            sort_by="publishedAt",
            page_size=5
        )
        
        if response and response.get("status") == "ok":
            articles = response.get("articles", [])
            headlines = [a["title"] for a in articles if a.get("title")]
            
            if not headlines:
                return "No recent news found."
                
            return "\n".join(f"- {h}" for h in headlines)
            
        return "No recent news found."
    except Exception as e:
        return f"Error fetching news for {company}: {e}"

def analyze_sentiment_with_langchain(ticker, stock_name, openai_api_key, news_api_key):
    """Uses LangChain to summarize news and determine sentiment."""
    # 1. Fetch News using the company name for better search results
    news_text = fetch_news(stock_name, news_api_key)
    
    if "No recent news found" in news_text or "Error" in news_text:
        return f"Could not retrieve news for analysis. Details: {news_text}"

    # 2. Initialize LLM
    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

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