from joblib import dump, load
import pandas as pd
import numpy as np
import os
import pandas_datareader as web
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import webbrowser

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

while(True):
    #Print menu and get ticker/date inputs
    print('Welcome to the Stock Predictor App. Please follow Instructions')
    print('1. Linear Regression Model ')
    print('2. LSTM Model ')
    print('3. Exit ')
    modelType = str(input ("Select an option: "))

    if(modelType == '1'):
        ticker = str(input ("Enter Stock Ticker Name: "))
        date=str(input ("Enter Date (yyyy-mm-dd): "))

        # Load linear regression model and stock data
        lr = load('models/models_%s.joblib'%ticker)
        df = pd.read_csv('data/%s.csv'%ticker, sep = ',', header = 0)

        # Get index of specified date and print close price for that day
        current_row=df.loc[df['Date'] == date]
        index=current_row.index.tolist()
        x_forecast = np.array(current_row.drop(['Date', 'Prediction'],1))
        print('\nPrice for %s'%date,':',x_forecast[0][3])

        # Predict close price of next day
        lr_prediction = lr.predict(x_forecast)
        print('Prediction for the 1 day out:', lr_prediction[0])

        # Get actual next day price
        actual=df.iloc[index[0]+1]
        print('Actual Price for 1 day out:',actual['Close'],'\n')

        # Display figure
        webbrowser.open('figs/%s.png'%ticker)


    elif(modelType == '2'):
        ticker = str(input ("Enter Stock Ticker Name: "))
        date=str(input ("Enter Date before 2020-10-01 (yyyy-mm-dd): "))

        # LSTM
        # Load in LSTM model
        lstm = load_model('lstm_models/' + ticker + '.h5')

        # Pull stock data for ticker
        df = web.DataReader(ticker, data_source = 'yahoo', start = '2015-01-01', end = '2020-10-01')
        data=df['Close']

        # Reshape data 
        dataset = np.array(data.values)
        dataset = np.reshape(dataset, (-1, 1))
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Get index of specified date
        df.reset_index(inplace=True,drop=False)
        current_row=df.loc[df['Date'] == date]
        index=current_row.index.tolist()

        # Get training data from scaled
        train_data = scaled_data[index[0]-60:index[0]]
        #train_data = np.array(train_data)
        train_data = np.array([train_data])
        # Reshape training data
        #np.reshape(train_data, shape = (1, 60, 1))

        # Get row for specified data and print close price
        x_forecast = np.array(current_row.drop(['Date'],1))
        print('\nPrice for %s'%date,':',x_forecast[0][3])

        # Predict close price for next day
        lstm_prediction = lstm.predict(train_data)
        lstm_prediction = scaler.inverse_transform(lstm_prediction)
        print('Prediction for the 1 day out:', lstm_prediction[0][0])

        # Get actual next day price
        actual=df.iloc[index[0]+1]
        print('Actual Price for 1 day out:',actual['Close'],'\n')

        # Display figure
        webbrowser.open('lstmfigs/%s.png'%ticker)

    elif (modelType == '3'):
        break

    else:
        print('Incorrect Input. Please enter a number 1-3.\n')