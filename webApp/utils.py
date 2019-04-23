import quandl
import pandas as pd
import numpy as np
import datetime

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def preprocess(tomorrow,scaler,look_back):
    #fetch data

    # data = quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
    # data.to_csv('btc.csv',sep='\t')
    #read data
    data = pd.read_csv('btc.csv',sep='\t')
    data = data.append({'Date':tomorrow},ignore_index = True)

    data['Weighted Price'].replace(0, np.nan, inplace=True)
    data['Weighted Price'].fillna(method='ffill', inplace=True)

    #normalize values in range [0,1]
    values = data['Weighted Price'].values.reshape(-1,1)
    values = values.astype('float32')

    scaled = scaler.fit_transform(values)

    #split data for training and testing
    train_size = int(len(scaled) * 0.80)
    test_size = len(scaled) - train_size
    train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
    print("Train length: ",len(train),"\tTest length: ",len(test))

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return data,trainX, trainY, testX, testY


def calcAccuracy(testY_inverse,ypredicted_inverse):
    predictionAccuracy = 0
    count = 0
    countZeroes = 0
    pctChangeAccuracy = 0
    for i  in range(len(ypredicted_inverse)-1):
        if(testY_inverse[i]!=0):
            actualPctChange = abs((testY_inverse[i]-testY_inverse[i+1])/testY_inverse[i] *100 )
            predictedPctChange = abs((ypredicted_inverse[i]-ypredicted_inverse[i+1])/ypredicted_inverse[i] *100 )
            if(actualPctChange != 0):
                pctChangeAccuracy+= abs((actualPctChange - predictedPctChange)/actualPctChange *100)
            if(testY_inverse[i] !=0):

                predictionAccuracy+= abs((testY_inverse[i][0] - ypredicted_inverse[i][0])/testY_inverse[i][0])
            if(count<10):
                # print(testY_inverse[i],":",ypredicted_inverse[i])
                # print(abs(((testY_inverse[i][0] - ypredicted_inverse[i][0])/testY_inverse[i][0])))
                # print(actualPctChange,':',predictedPctChange,':',pctChangeAccuracy)
                count+=1
    pctChangeAccuracy = abs(pctChangeAccuracy/len(ypredicted_inverse))/100
    accuracy = (predictionAccuracy/len(ypredicted_inverse))*100
    return (100-accuracy),(100-pctChangeAccuracy)
