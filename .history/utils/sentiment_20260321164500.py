import os
import numpy as np
from newsapi import NewsApiClient
from transformers import pipeline

API_KEY = os.getenv("NEWS_API_KEY")

newsapi = NewsApiClient(api_key=API_KEY)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

COMPANY_NAMES = {
    "HDFC BANK": "HDFC Bank",
    "TCS": "Tata Consultancy Services",
    "RELIANCE": "Reliance Industries",
    "HINDUSTAN UNILEVER": "Hindustan Unilever",
    "SUN PHARMA": "Sun Pharmaceutical"
}

def fetch_news(company):

    response = newsapi.get_everything(
        q=company,
        language="en",
        sort_by="publishedAt",
        page_size=10
    )

    headlines = [
        a["title"] for a in response["articles"]
        if a.get("title")
    ]

    return headlines

def sentiment_score(headlines):

    if len(headlines) == 0:
        return 0

    scores = []

    for text in headlines:
        result = sentiment_model(text[:512])[0]

        if result["label"] == "positive":
            scores.append(result["score"])
        elif result["label"] == "negative":
            scores.append(-result["score"])

    return np.mean(scores)