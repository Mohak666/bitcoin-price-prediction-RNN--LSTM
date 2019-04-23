from flask import Flask
from flask import render_template
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
import quandl
from utils import create_dataset,preprocess, calcAccuracy

app = Flask(__name__)

lstm = load_model('./models/model-lstm.hdf5')
gru = load_model('./models/model-gru.hdf5')
# data = pd.read_csv('btc.csv',sep='\t')

tomorrow = datetime.datetime.today().strftime("%Y-%m-%d")
look_back = 10
deficit = 1
scaler = MinMaxScaler(feature_range=(0, 1))


data,trainX, trainY,testX,testY = preprocess(tomorrow,scaler,look_back)

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
    return render_template('index.html',tomorrow = nextDayPrice,today = price)

@app.route('/results')
def results():
    return render_template('accuracy.html',lAcc =lAcc,lPctAcc=lPctAcc[0],gAcc=gAcc,gPctAcc=gPctAcc[0])


if __name__ == '__main__':
    app.run()
