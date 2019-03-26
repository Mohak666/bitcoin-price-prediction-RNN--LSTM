from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM,Activation
import numpy as np

# model = Sequential()
#
# model.add(LSTM(50,
#     return_sequences=True))
# model.add(Dropout(0.35))
#
# model.add(LSTM(
#     100,
#     return_sequences=False))
# model.add(Dropout(0.35))
#
# model.add(Dense(1))
# model.add(Activation('relu'))

model = load_model('./models/model.hdf5')
