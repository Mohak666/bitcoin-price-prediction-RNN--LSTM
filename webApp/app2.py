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
from utils import create_dataset,preprocess, calcAccuracy, prediction
from train import training

app = Flask(__name__)

nextDayPrice,price,lAcc,lPctAcc, gAcc,gPctAcc= prediction()

@app.route('/')
def hello_world():
    # accuracy,pctChangeAccuracy,nextDayPrice,price = run_model()
    return render_template('index.html',tomorrow = nextDayPrice,today = price)

@app.route('/results')
def results():
    return render_template('accuracy.html',lAcc =lAcc,lPctAcc=lPctAcc[0],gAcc=gAcc,gPctAcc=gPctAcc[0])


if __name__ == '__main__':
    app.run()
