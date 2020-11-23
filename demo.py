from joblib import dump, load
import pandas as pd
import numpy as np

ticker = str(raw_input ("Enter Stock Ticker Name: "))
date=str(raw_input ("Enter Date (yyyy-mm-dd): "))
lr = load('models/models_%s.joblib'%ticker)
df = pd.read_csv('data/%s.csv'%ticker, sep = ',', header = 0)

current_row=df.loc[df['Date'] == date]
index=current_row.index.tolist()
x_forecast = np.array(current_row.drop(['Date', 'Prediction'],1))
print('Current price:',x_forecast[0][3])
lr_prediction = lr.predict(x_forecast)
print('Prediction for the 1 day out:', lr_prediction[0])
actual=df.iloc[index[0]+1]
print('Actual Price for 1 day out:',actual['Close'])