import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas_datareader
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import OrderedDict
from joblib import dump, load 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def genLRPlotModel(ticker):
    df = pd.read_csv('data/%s.csv'%ticker, sep = ',', header = 0)

    X = np.array(df['Close'])

    y = np.array(df['Prediction'])

    n = 30 # number of days on which predictions are made

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
    actual = df.drop(['Date', 'Prediction'], 1)[-n:]
    actual['Predictions'] = lr_prediction

    df['lr_prediction']=df['Prediction']
    df['lr_prediction'][-n:]=lr_prediction
    df.to_csv('predictions/{}.csv'.format(ticker))

    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(df['Close'])
    plt.plot(actual[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    #plt.show()

    plt.savefig('figs/%s.png'%ticker)
    plt.close()

    return lr_confidence
    
################################### - LSTM - ##################################################
def trainLSTM(ticker):
    #Create a new Dataframe
    df = pandas_datareader.DataReader('V', data_source = 'yahoo', start = '2015-01-01', end = '2020-10-01')
    data = df[['Close']]
    n = 60

    #Convert to numpy array
    dataset = np.array(data.values)
    dataset = np.reshape(dataset, (-1, 1))
    
    #Get the number of rows to train the model
    training_data_len = math.ceil(len(dataset) * .8)

    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    #Split the data into x_train and y_train
    x_train = []
    y_train = []
    for i in range(n, len(train_data)):
        x_train.append(train_data[i-n:i, 0])
        y_train.append(train_data[i, 0])
    
    #Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data
    # (#samples, timesteps, and features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Build the LSTM model
    model = Sequential()
    # 50 neurons, (timesteps, features)
    model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss ='mean_squared_error')

    model.fit(x_train, y_train, batch_size = 1, epochs = 5)

    #Create the testing dataset
    #Create a new array containing scaled values from index size-n to size
    # [last n values, all the columns]
    test_data = scaled_data[training_data_len - n:, :]

    #Create the datasets x_test, y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(n, len(test_data)):
        # Past n values
        x_test.append(test_data[i-n:i, 0])

    #convert to numpy array
    x_test = np.array(x_test)

    #Reshape the data for the LSTM model to 3-D
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get the models predicted price and values
    predictions = model.predict(x_test)
    # We want predictions to contain the same values as y_test dataset
    predictions = scaler.inverse_transform(predictions)

    file_name = 'lstm_models/' + ticker + '.h5'
    model.save(file_name)

    #Get the root mean squred error (RMSE) - lower the better
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print("LSTM RMSE for %s: "%ticker, rmse)

    ################################### - LSTM - ##################################################

def plotLSTM(ticker):
    #Create a new Dataframe
    n = 60
    df = pandas_datareader.DataReader(ticker, data_source = 'yahoo', start = '2015-01-01', end = '2020-10-01')
    data = df[['Close']]

    dataset = np.array(data.values)
    dataset = np.reshape(dataset, (-1, 1))

    #Get the number of rows to train the model
    training_data_len = math.ceil(len(data) * .8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training data set
    train_data = scaled_data[0:training_data_len, 0:4]
    #Split the data into x_train and y_train
    x_train = []
    y_train = []
    for i in range(n, len(train_data)):
        x_train.append(train_data[i-n:i, 0:4])
        y_train.append(train_data[i, :])

    x_train, y_train = np.array(x_train), np.array(y_train)

    test_data = scaled_data[training_data_len - n:, :]
    #Create the datasets x_test, y_test
    x_test = []
    # 61st values
    y_test = dataset[training_data_len:, :]
    for i in range(n, len(test_data)):
        # Past n values
        x_test.append(test_data[i-n:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    file_name = 'lstm_models/' + ticker + '.h5'
    model = load_model(file_name)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    #train = data.loc[:'2020-09-01']
    #valid = data.loc['2020-09-01':]
    #print(data.loc[:'2020-09-01'])
    valid['Predictions'] = predictions
    #print(type(train))
    #print(type(valid))
    #predictions_series = []
    #indices = list(range(162, 202))
    #for prediction in predictions:
    #    predictions_series.append(prediction)
    #predictions = pd.Series(predictions_series, index = indices)

    # Visulaize the date
    plt.figure(figsize=(16,8))
    plt.title(ticker.upper())
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price USD ($)', fontsize=12)
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
    plt.savefig('lstm_figs/' + ticker + '.png')
    #plt.show()


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

    confTot = 0

    """ plotLSTM('aapl')
    plotLSTM('msft')
    plotLSTM('v') """
    for stock in list(dow_jones_dict.keys()):
        confCurr = genLRPlotModel(stock)
        #trainLSTM(stock)
        confTot += confCurr
    
    #confAvg = confTot / 31

    #print("Average lr confidence is: ", confAvg)