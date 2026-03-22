import yfinance as yf
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def get_stock_news(ticker):
    """Fetches recent news headlines securely using Yahoo Finance RSS."""
    try:
        # Fetch directly from Yahoo's official RSS feed
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        # Parse the XML to extract titles
        root = ET.fromstring(xml_data)
        headlines = []
        
        # Grab the top 5 news items
        for item in root.findall('.//item')[:5]:
            title_element = item.find('title')
            if title_element is not None and title_element.text:
                headlines.append(title_element.text)
                
        if not headlines:
            return "No recent news found."
            
        return "\n".join(f"- {h}" for h in headlines)
        
    except Exception as e:
        return f"Error fetching news from RSS feed: {e}"

def analyze_sentiment_with_langchain(ticker, stock_name, api_key):
    """Uses LangChain and Gemini to summarize news and determine sentiment."""
    # 1. Fetch News
    news_text = get_stock_news(ticker)
    
    if "No recent news found" in news_text or "Error" in news_text:
        return f"Could not retrieve news for analysis. Details: {news_text}"

    try:
        # 2. Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            temperature=0.2, 
            model="gemini-2.5-flash", 
            api_key=api_key
        )

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
        
    except Exception as e:
        return f"Generative AI Error: Please verify your Google API Key and try again. Error details: {e}"