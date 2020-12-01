import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from collections import OrderedDict
from joblib import dump, load 

def genPlotModel(ticker):
    df = pd.read_csv('data/%s.csv'%ticker, sep = ',', header = 0)

    df.shape
    """
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    #plt.show()
    plt.close()
    """

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
    dump(lr, 'models/models_%s.joblib'%ticker)

    lr_confidence = lr.score(x_test1, y_test)
    print("lr confidence for %s: "%ticker, lr_confidence)

    x_forecast = np.array(df.drop(['Date', 'Prediction'],1))[-1:]
    lr_prediction = lr.predict(x_forecast)
    print('Prediction for the 1 day out:', lr_prediction)

    lr_prediction = lr.predict(x_test1)
    train = x_train1
    valid = df.drop(['Date', 'Prediction'],1)[-n:]
    valid['Predictions'] = lr_prediction

    df['lr_prediction']=df['Prediction']
    df['lr_prediction'][-n:]=lr_prediction
    df.to_csv('predictions/{}.csv'.format(ticker))

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

    return lr_confidence

if __name__ == '__main__':
    dow_jones_dict = OrderedDict()
    dow_jones_dict['aapl'] = 'Apple'
    dow_jones_dict['amgn'] = 'Amgen'
    dow_jones_dict['axp'] = 'American Express'
    dow_jones_dict['ba'] = 'Bank of America'
    dow_jones_dict['cat'] = 'Caterpillar Inc.'
    dow_jones_dict['crm'] = 'Salesforce'
    dow_jones_dict['csco'] = 'Cisco Systems'
    dow_jones_dict['cvx'] = 'Chevron Corporation'
    dow_jones_dict['dis'] = 'Disney'
    dow_jones_dict['^dji'] = 'Dow Jones Index'
    dow_jones_dict['dow'] = 'Dow Inc.'
    dow_jones_dict['gs'] = 'Goldman Sachs'
    dow_jones_dict['hd'] = 'The Home Depot'
    dow_jones_dict['hon'] = 'Honeywell'
    dow_jones_dict['ibm'] = 'IBM'
    dow_jones_dict['intc'] = 'intel'
    dow_jones_dict['jnj'] = 'Johnson & Johnson'
    dow_jones_dict['jpm'] = 'JPMorgan Chase'
    dow_jones_dict['ko'] = 'Coca-Cola'
    dow_jones_dict['mcd'] = "McDonald's"
    dow_jones_dict['mmm'] = '3M'
    dow_jones_dict['mrk'] = 'Merck & Co.'
    dow_jones_dict['msft'] = 'Microsoft'
    dow_jones_dict['nke'] = 'Nike'
    dow_jones_dict['pg'] = 'Procter & Gamble'
    dow_jones_dict['trv'] = 'The Travelers Companies'
    dow_jones_dict['unh'] = 'UnitedHealth Group'
    dow_jones_dict['v'] = 'Visa'
    dow_jones_dict['vz'] = 'Verizon'
    dow_jones_dict['wba'] = 'Walgreens'
    dow_jones_dict['wmt'] = 'Walmart'

    confTot=0

    for stock in list(dow_jones_dict.keys()):
        confCurr=genPlotModel(stock)
        confTot+=confCurr
    
    confAvg=confTot/31

    print("Average lr confidence is: ", confAvg)