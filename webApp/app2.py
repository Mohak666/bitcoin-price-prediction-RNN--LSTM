from flask import Flask
from flask import render_template
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, save_model,load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import numpy as np
import seaborn as sns
import h5py

app = Flask(__name__)

lstm = load_model('./models/model-lstm.hdf5')
gru = load_model('./models/model-gru.hdf5')
# data = pd.read_csv('btc.csv',sep='\t')

tomorrow = datetime.datetime.today().strftime("%Y-%m-%d")
look_back = 10
deficit = 1
scaler = MinMaxScaler(feature_range=(0, 1))

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def preprocess():

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


data,trainX, trainY,testX,testY = preprocess()

LSTMpredicted = lstm.predict(testX)
LSTMpredicted = LSTMpredicted[deficit:]
LSTMpredicted_inverse = scaler.inverse_transform(LSTMpredicted.reshape(-1, 1))

GRUpredicted = gru.predict(testX)
GRUpredicted = GRUpredicted[deficit:]
GRUpredicted_inverse = scaler.inverse_transform(GRUpredicted.reshape(-1, 1))

testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))


lAcc,lPctAcc =calcAccuracy(testY_inverse,LSTMpredicted_inverse)
gAcc,gPctAcc = calcAccuracy(testY_inverse,GRUpredicted_inverse)
print(lAcc,lPctAcc)
print(gAcc,gPctAcc)
testY_inverse = testY_inverse[:-deficit]

predictDates = data['Date'].tail(len(LSTMpredicted_inverse))

testY_reshape = testY_inverse.reshape(len(LSTMpredicted_inverse))
LSTMpredicted_reshape = LSTMpredicted_inverse.reshape(len(LSTMpredicted_inverse))
GRUpredicted_reshape = GRUpredicted_inverse.reshape(len(GRUpredicted_inverse))


sns.lineplot(x = predictDates.index, y = testY_reshape, label = 'Actual Price')
fig1 = sns.lineplot(x=predictDates.index, y=LSTMpredicted_reshape, label= 'LSTM Predicted Price')
image1 = fig1.get_figure()
image1.savefig('./static/lstm.png',dpi = 400, overwrite = True)

plt.clf()

sns.lineplot(x = predictDates.index, y = testY_reshape, label = 'Actual Price')
fig2 = sns.lineplot(x=predictDates.index, y=GRUpredicted_reshape, label= 'GRU Predicted Price')
image2 = fig2.get_figure()
image2.savefig('./static/gru.png',dpi = 400, overwrite = True)

nextDayPrice = GRUpredicted_reshape[-1]
price = testY_reshape[-2]

@app.route('/')
def hello_world():
    # accuracy,pctChangeAccuracy,nextDayPrice,price = run_model()
    return render_template('front.html',tomorrow = nextDayPrice,today = price)

@app.route('/results')
def results():
    return render_template('front2.html',lAcc =lAcc,lPctAcc=lPctAcc,gAcc=gAcc,gPctAcc=gPctAcc)


if __name__ == '__main__':
    app.run()
