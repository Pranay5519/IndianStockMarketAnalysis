# 📈 FinVise AI: Indian Stock Market Analysis Module

**Author:** Pranay  
**Role:** Full Stack AI/ML Engineer 

## 📝 Project Overview
This repository contains a Full Stack AI/ML dashboard designed to analyze and predict trends within the Indian equity market (NSE). The application goes beyond standard technical indicators by integrating machine learning ensembles, unsupervised anomaly detection, and generative AI sentiment analysis.

## 🚀 Core Features & AI/ML Architecture

### 1. Market Analysis & Predictive Modeling
* **Interactive Visualization:** Dynamic Plotly OHLCV charts with volume tracking and selectable timeframes.
* **Algorithmic Support/Resistance:** Programmatically calculates local minima and maxima using rolling window extrema detection to dynamically plot S/R lines.
* **Unsupervised Anomaly Detection:** An `IsolationForest` model evaluates rolling volatility, volume spikes, and returns to flag and plot anomalous trading days directly on the chart.
* **Stacking Regressor Pipeline:** A predictive ensemble utilizing **LightGBM** and **Random Forest** base estimators, with a Ridge regression meta-model, trained on trailing technical indicators to forecast next-day closing prices.

### 2. Generative AI News Sentiment
* **Real-time News Ingestion:** Integrates `NewsAPI` to fetch live, structured headlines for selected equities.
* **LangChain NLP Engine:** Feeds live headlines into a LangChain prompt template connected to Google's **Gemini 2.5 Flash LLM**. 
* **Automated Insights:** Extracts a cohesive market narrative, classifies overall sentiment (Bullish/Bearish/Neutral), and identifies potential risk catalysts.

## 🧰 Tech Stack
* **Frontend:** Streamlit
* **Data Ingestion:** yfinance, NewsAPI
* **Machine Learning:** Scikit-Learn, LightGBM, ta (Technical Analysis)
* **Generative AI:** LangChain, Google Gemini API
* **Visualization:** Plotly

## 💻 Local Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME