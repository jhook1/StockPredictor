import os
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# News Data is scraped from https://seekingalpha.com/symbol/<ticker>/analysis or Google News, Date Range from 01/31/2020 - 11/17/2020
def get_stock(ticker, company, method = 'seeking alpha'):
    sentiments = {}
    ticker_data = yf.Ticker(ticker)
    data = ticker_data.history(start = '2020-1-30', end = '2020-11-17')
    data = data.drop(['Dividends', 'Stock Splits'], axis = 1)
    data = data.assign(Sentiment = 0)
    if(method == 'seeking alpha'):
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
    elif(method == 'gn'):
        googlenews = GoogleNews(start = '01/30/2020', end = '11/17/2020')
        googlenews.search(company)
        for i in range(2, 6):
            googlenews.getpage(i)
        results = googlenews.result()
        for result in results:
            headline_sentiment = analyzer.polarity_scores(result['title'])['compound']
            try:
                article_date = pd.to_datetime(result['date'], format = '%b %d, %Y')
            except:
                continue
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
    dow_jones_stocks = {
        'aapl' : 'Apple', 
        'amgn' : 'Amgen', 
        'axp' : 'American Express', 
        'ba' : 'Bank of America', 
        'cat' : 'Caterpillar Inc', 
        'crm' : 'Salesforce', 
        'csco' : 'Cisco Systems',
        'cvx' : 'Chevron Corporation', 
        'dis' : 'Disney', 
        '^dji' : 'Dow Jones Index',
        'dow' : 'Dow Inc.', 
        'gs' : 'Goldman Sachs', 
        'hd' : 'The Home Depot', 
        'hon' : 'Honeywell', 
        'ibm' : 'IBM', 
        'intc' : 'Intel',
        'jnj' : 'Johnson & Johnson', 
        'jpm' : 'JPMorgan Chase', 
        'ko' : 'Coca-Cola', 
        'mcd' : "McDonald's", 
        'mmm' : '3M', 
        'mrk' : 'Merck & Co.', 
        'msft' : 'Microsoft', 
        'nke' : 'Nike',
        'pg' : 'Procter & Gamble', 
        'trv' : 'The Travelers Companies', 
        'unh' : 'UnitedHealth Group', 
        'v' : 'Visa', 
        'vz' : 'Verizon', 
        'wba' : 'Walgreens', 
        'wmt' : 'Walmart'
    }

    for stock in list(dow_jones_stocks.keys()):
        get_stock(stock, dow_jones_stocks[stock], method = 'gn')
        print(stock + ' Data Generated')