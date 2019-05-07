# from utils import create_dataset, preprocess, calcAccuracy
import quandl
import pandas as pd
import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense,Dropout,GRU,Activation
import numpy as np
gPctAcc = 0
def training():

    look_back = 10
    tomorrow = datetime.datetime.today().strftime("%Y-%m-%d")
    deficit = 1
    scaler = MinMaxScaler(feature_range=(0, 1))

    data,trainX, trainY,testX,testY = preprocess(tomorrow,scaler,look_back,0)


    gru = Sequential()
    gru.add(GRU(50,input_shape = trainX[0].shape,
        return_sequences=True))
    gru.add(Dropout(0.35))

    gru.add(GRU(
        100,
        return_sequences=False))
    gru.add(Dropout(0.35))

    gru.add(Dense(1))
    gru.add(Activation('relu'))

    opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-5)
    gru.compile(loss='mae', optimizer=opt,metrics=['accuracy','mse'])
    history = gru.fit(trainX, trainY, epochs=500, batch_size=1750, validation_data=(testX, testY), verbose=1, shuffle=False)
    save_model(gru,'./models/model-gru.hdf5',overwrite = True,include_optimizer = True)
    # gru_score = gru.evaluate(testX,testY,verbose = 1)
    # print(gru.metrics_names)
    # print("score: ",score)

    GRUpredicted = gru.predict(testX)
    GRUpredicted = GRUpredicted[deficit:]
    GRUpredicted_inverse = scaler.inverse_transform(GRUpredicted.reshape(-1, 1))

    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

    gAcc,gPctAcc = calcAccuracy(testY_inverse,GRUpredicted_inverse)

    print(gAcc,gPctAcc)
    return gPctAcc

# training()
