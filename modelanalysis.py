import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def genPlotModel(ticker):
    df = pd.read_csv('data/%s.csv'%ticker, sep = ',', header = 0)

    df.shape
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    #plt.show()
    plt.close()

    X = np.array(df['Close'])

    y = np.array(df['Prediction'])

    n=30 # number of days on which predictions are made

    x_train1 = np.array(df.drop(['Date', 'Prediction'],1))[:-n]
    x_test1 = np.array(df.drop(['Date', 'Prediction'],1))[-n:]
    x_train = X[:-n]
    x_test = X[-n:]
    y_train = y[:-n]
    y_test = y[-n:]

    lr = LinearRegression()
    lr.fit(x_train1, y_train)

    lr_confidence = lr.score(x_test1, y_test)
    print("lr confidence: ", lr_confidence)

    x_forecast = np.array(df.drop(['Date', 'Prediction'],1))[-1:]
    lr_prediction = lr.predict(x_forecast)
    print('Prediction for the 1 day out:', lr_prediction)

    lr_prediction = lr.predict(x_test1)
    train = x_train1
    valid = df.drop(['Date', 'Prediction'],1)[-n:]
    valid['Predictions'] = lr_prediction
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    #plt.show()

    plt.savefig('figs/%s.png'%ticker)
    plt.close()

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
        genPlotModel(stock)