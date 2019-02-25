import numpy as np
import pandas as pd
import random
from tensorflow.keras.layers import LSTM
from datetime import datetime
from sklearn import preprocessing
from collections import deque

futurePredictPeriod = 3
seqLen = 60
# function to convert given date to unix time
def convertToUnixTime(date):
    date = date.split(',')
    date  = (' ').join(date)
    date = datetime.strptime(date,'%b %d %Y')
    return int(date.strftime('%s'))

#function to convert unix time to normal date
def convertToNormalTime(date):
    return datetime.fromtimestamp(date)


def removeCommas(value):
    value = value.split(',')

    return int(('').join(value))



def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess(dataset):
    dataset  = dataset.drop('Future',1)

    for col in dataset.columns:
        if col!= 'Target' and col!= 'Date':
            dataset[col] = dataset[col].apply(lambda x: int(x))
            dataset[col] = dataset[col].pct_change()
            dataset.dropna(inplace = True)
            dataset[col] = preprocessing.scale(dataset[col].values)
    dataset.dropna(inplace = True)

    sequential_data = []
    prev_days = deque(maxlen = seqLen)
    # print(dataset.head(15))
    # print(dataset.values[0])
    for i in dataset.values:
        prev_days.append([n for n in i[:-1]])#leave out target column
        if len(prev_days) == seqLen:
            sequential_data.append([np.array(prev_days),i[-1]])
            # print(sequential_data)
    random.shuffle(sequential_data)

    # buys = []
    # sells = []
    #
    # for seq,target in sequential_data:
    #     if target == 0:
    #         sells.append([seq,target])
    #     else:
    #         buys.append([seq,target])
    #
    # lower = min(len(buys),len(sells))
    #
    # sells = sells[:lower]
    # buys = buys[:lower]
    #
    # sequential_data = buys+sells
    # random.shuffle(sequential_data)

    x = []
    y = []

    for seq,target in sequential_data:
        x.append(seq)
        y.append(target)
    return np.array(x),y

    # print(len(sequential_data))
    # return dataset



#read dataset from a csv file
dataset = pd.read_csv('bitcoin_price.csv')
dataset = dataset[dataset['Volume']!= '-']
dataset.drop(['Open','High','Low','Market Cap'],axis = 1,inplace = True)
print(dataset.head())



#convert date for every entry into unix time
dataset['Date'] = dataset['Date'].apply(convertToUnixTime)
dataset['Future'] = dataset['Close'].shift(-futurePredictPeriod)


# print(dataset['Date'].apply(convertToNormalTime))
dataset['Target'] = list(map(classify,dataset['Close'],dataset['Future']))

dataset['Volume'] = dataset['Volume'].apply(removeCommas)


x,y = preprocess(dataset)


trainLen = int(len(x)*0.85)
trainX,trainY = x[:trainLen],y[:trainLen]
validationX,validationY = x[trainLen:],y[trainLen:]

print(len(trainX),len(validationX))
print(trainX[0])
print(validationX[0])
