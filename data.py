import os
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_stock(ticker):
    sentiments = {}
    ticker_data = yf.Ticker(ticker)
    data = ticker_data.history(start = '2020-1-30', end = '2020-8-2')
    data = data.drop(['Dividends', 'Stock Splits'], axis = 1)
    data = data.assign(Sentiment = 0)
    soup = BeautifulSoup(open('html/{}.txt'.format(ticker)), 'html.parser')
    articles = soup.find_all('article')
    for article in articles:
        article_title = article.find_all('a')[1].text
        spans = article.find_all('span')
        if(len(spans) == 1):
            article_date = spans[0].text
        else:
            article_date = spans[1].text
        article_date = article_date.split(', ')[1].replace('.', '') + ' 2020'
        article_date = pd.to_datetime(article_date, format = '%b %d %Y')
        headline_sentiment = analyzer.polarity_scores(article_title)['compound']
        if(article_date not in sentiments.keys()):
            sentiments[article_date] = [headline_sentiment]
        else:
            sentiments[article_date].append(headline_sentiment)
    data['Prediction'] = data[['Close']].shift(-1)
    data = data[:-1]
    for s in sentiments:
        average_sentiment = np.average(sentiments[s])
        if(s in data.index):
            data.loc[s, 'Sentiment'] = average_sentiment
    data.to_csv('data/{}.csv'.format(ticker))

if __name__ == '__main__':
    dow_jones_stocks = ['aapl', 'amgn', 'axp', 'ba', 'cat', 'crm', 'csco',
    'cvx', 'dis', 'dow', 'gs', 'hd', 'hon', 'ibm', 'intc',
    'jnj', 'jpm', 'ko', 'mcd', 'mmm', 'mrk', 'msft', 'nke',
    'pg', 'trv', 'unh', 'v', 'vz', 'wba', 'wmt']

    for stock in dow_jones_stocks:
        get_stock(stock)