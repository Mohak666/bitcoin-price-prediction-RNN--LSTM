import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM
from datetime import datetime

# function to convert given date to unix time
def convertToUnixTime(date):
    date = date.split(',')
    date  = (' ').join(date)
    date = datetime.strptime(date,'%b %d %Y')
    return int(date.strftime('%s'))

#function to convert unix time to normal date
def convertToNormalTime(date):
    return datetime.fromtimestamp(date)


#read dataset from a csv file
dataset = pd.read_csv('bitcoin_price.csv')
print(dataset.head())

#convert date for every entry into unix time
dataset['Date'] = dataset['Date'].apply(convertToUnixTime)
print(dataset[['Date','Volume','Close']].head())

# print(dataset['Date'].apply(convertToNormalTime))
