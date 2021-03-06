from flask import Flask
from flask import render_template
from math import sqrt
from matplotlib import pyplot
import pandas as pd
import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import numpy as np
import seaborn as sns
import h5py

app = Flask(__name__)

# def run_model():
# import quandl
# data = quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
# data.to_csv('btc.csv',sep='\t')
tomorrow = datetime.datetime.today().strftime("%Y-%m-%d")
look_back = 10

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

deficit = 1

data = pd.read_csv('btc.csv',sep='\t')
data = data.append({'Date':tomorrow},ignore_index = True)
print(data.head())
print(data.tail())


sns.lineplot(x = data.index, y = data['Weighted Price'])

#fill missing values with previous day values
data['Weighted Price'].replace(0, np.nan, inplace=True)
data['Weighted Price'].fillna(method='ffill', inplace=True)

sns.lineplot(x = data.index, y = data['Weighted Price'])

#normalize values in range [0,1]
values = data['Weighted Price'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

#split data for training and testing
train_size = int(len(scaled) * 0.80)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print("Train length: ",len(train),"\tTest length: ",len(test))

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# print(trainX[:5])
# print(trainY[:5])

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX[0])

print("shape: ",trainX.shape)
model = Sequential()

model.add(LSTM(50,input_shape = trainX[0].shape,
    return_sequences=True))
model.add(Dropout(0.35))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.35))

model.add(Dense(1))
model.add(Activation('relu'))

opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-5)
model.compile(loss='mae', optimizer=opt,metrics=['accuracy','mse'])
history = model.fit(trainX, trainY, epochs=500, batch_size=1750, validation_data=(testX, testY), verbose=0, shuffle=False)

score = model.evaluate(testX,testY,verbose = 1)
print(model.metrics_names)
print("score: ",score)
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
pyplot.show()

# model.save('model1.hdf5')
save_model(model,'./models/model.hdf5',overwrite = True,include_optimizer = True)
# model.save_weights("model_weights.h5")

# pyplot.plot(ypredicted, label='predict')
# pyplot.plot(testY, label='true')
# pyplot.legend()
# pyplot.show()
ypredicted = model.predict(testX)
ypredicted = ypredicted[deficit:]
ypredicted_inverse = scaler.inverse_transform(ypredicted.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
testY_inverse = testY_inverse[:-deficit]



predictionAccuracy = 0
count = 0
countZeroes = 0
pctChangeAccuracy = 0
for i  in range(len(ypredicted_inverse)-1):
    actualPctChange = abs((testY_inverse[i]-testY_inverse[i+1])/testY_inverse[i] *100 )
    predictedPctChange = abs((ypredicted_inverse[i]-ypredicted_inverse[i+1])/ypredicted_inverse[i] *100 )
    if(actualPctChange != 0):
        pctChangeAccuracy+= abs((actualPctChange - predictedPctChange)/actualPctChange *100)
    predictionAccuracy+= abs((testY_inverse[i][0] - ypredicted_inverse[i][0])/testY_inverse[i][0])
    if(count<10):
        # print(testY_inverse[i],":",ypredicted_inverse[i])
        # print(abs(((testY_inverse[i][0] - ypredicted_inverse[i][0])/testY_inverse[i][0])))
        # print(actualPctChange,':',predictedPctChange,':',pctChangeAccuracy)
        count+=1
pctChangeAccuracy = abs(pctChangeAccuracy/len(ypredicted_inverse))/100
accuracy = (predictionAccuracy/len(ypredicted_inverse))*100
print("\nprediction accuracy = ",100-accuracy,"%")
print("\npercent change accuracy = ",100-pctChangeAccuracy,"%")


# pyplot.plot(ypredicted_inverse, label='predict')
# pyplot.plot(testY_inverse, label='actual', alpha=0.5)
# pyplot.legend()


predictDates = data['Date'].tail(len(ypredicted_inverse))

testY_reshape = testY_inverse.reshape(len(ypredicted_inverse))
ypredicted_reshape = ypredicted_inverse.reshape(len(ypredicted_inverse))



sns.lineplot(x = predictDates.index, y = testY_reshape, label = 'Actual Price')
fig = sns.lineplot(x=predictDates.index, y=ypredicted_reshape, label= 'Predict Price')
image = fig.get_figure()
image.savefig('./static/actual-vs-predicted.png',dpi = 400, overwrite = True)
pyplot.show()

nextDayPrice = ypredicted_reshape[-1]
price = testY_reshape[-2]

# nextDayPrice = model.predict(testX[[-1]])
# nextDayPrice  = scaler.inverse_transform(price.reshape(-1, 1))
print("Forecasted price for tomorrow: ", nextDayPrice)
# return accuracy,pctChangeAccuracy,nextDayPrice,today

t=pyplot.show()
@app.route('/')
def hello_world():
    # accuracy,pctChangeAccuracy,nextDayPrice,price = run_model()
    return render_template('front.html',p=100-accuracy,pct = 100- pctChangeAccuracy, tomorrow = nextDayPrice,today = price)




if __name__ == '__main__':
    app.run()
