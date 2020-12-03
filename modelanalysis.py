import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from collections import OrderedDict
from joblib import dump, load 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math

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
    
    ################################### - LSTM - ##################################################

    #Create a new Dataframe
    data = df['Close']
    # print(type(data))
    #Convert to numpy array
    dataset = np.array(data.values)
    dataset = np.reshape(dataset, (-1, 1))
    #print(dataset)
    #Get the number of rows to train the model
    training_data_len = math.ceil(len(dataset) * .8)

    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    #print(scaled_data)

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
    x_train.shape

    #Build the LSTM model
    model = Sequential()
    # 50 neurons, (timesteps, features)
    model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss ='mean_squared_error')

    model.fit(x_train, y_train, batch_size = 1, epochs=1)

    #Create the testing dataset
    #Create a new array containing scaled values from index size-n to size
    # [last n values, all the columns]
    test_data = scaled_data[training_data_len - n:, :]
    #Create the datasets x_test, y_test
    x_test = []
    # 61st values
    y_test = dataset[training_data_len:, :]
    for i in range(n, len(test_data)):
        # Past n values
        x_test.append(test_data[i-n:i, 0])

    #convert to numpy array
    x_test = np.array(x_test)
    # print(x_test.shape)

    #Reshape the data for the LSTM model to 3-D
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Get the models predicted price and values
    predictions = model.predict(x_test)
    # We want predictions to contain the same values as y_test dataset
    predictions = scaler.inverse_transform(predictions)
    # print(type(predictions))

    #Get the root mean squred error (RMSE) - lower the better
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print("LSTM RMSE for %s: "%ticker, rmse)

    ################################### - LSTM - ##################################################


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