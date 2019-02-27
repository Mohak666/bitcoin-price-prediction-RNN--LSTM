from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns

# import quandl
# data = quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
# data.to_csv('btc.csv',sep='\t')
look_back = 2

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


data = pd.read_csv('btc.csv',sep='\t')
# print(data.head())
# print(data.tail())

# btc_trace = go.Scatter(x=data.index, y=data['Weighted Price'], name= 'Price')
# py.iplot([btc_trace])
# print(type(btc_trace))
sns.lineplot(x = data.index, y = data['Weighted Price'])

#fill missing values with previous day values
data['Weighted Price'].replace(0, np.nan, inplace=True)
data['Weighted Price'].fillna(method='ffill', inplace=True)

sns.lineplot(x = data.index, y = data['Weighted Price'])

#normalize valuesin range [0,1]
values = data['Weighted Price'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

#split data for training and testing
train_size = int(len(scaled) * 0.8)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print("Train length: ",len(train),"\tTest length: ",len(test))


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# print(trainX[:5])
# print(trainY[:5])

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='rmsprop',metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(testX)
pyplot.plot(yhat, label='predict')
pyplot.plot(testY, label='true')
pyplot.legend()
pyplot.show()
# score = model.evaluate(testY, yhat, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

predictionAccuracy = 0
count = 0
for i  in range(len(testY_inverse)):
    # print(testY_inverse[i],":",yhat_inverse[i])
    predictionAccuracy+= abs((testY_inverse[i][0] - yhat_inverse[i][0])/testY_inverse[i][0])
    if(count<30):
        print(abs(((testY_inverse[i][0] - yhat_inverse[i][0])/testY_inverse[i][0])))
        count+=1
accuracy = (predictionAccuracy/len(testY_inverse))*100
print("\nprediction accuracy = ",100-accuracy,"%")


rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)


pyplot.plot(yhat_inverse, label='predict')
pyplot.plot(testY_inverse, label='actual', alpha=0.5)
pyplot.legend()


predictDates = data['Date'].tail(len(testX))

testY_reshape = testY_inverse.reshape(len(testY_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))

# actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name= 'Actual Price')
# predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')
# py.iplot([predict_chart, actual_chart])
sns.lineplot(x = predictDates, y = testY_reshape, label = 'Actual Price')
sns.lineplot(x=predictDates, y=yhat_reshape, label= 'Predict Price')
pyplot.show()
