from markethistory import MarketHistory
from constants import * 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

CLOSE_INDEX = 0 ###USED TO CALCULATE RETURNS###
DEFAULT_PARAMS = {'max_depth':5, 'eta':.25, 'silent':0,'alpha':1,'min_child_weight':5, 'objective':"binary:logistic" }

class Agent:
    """
    Train a different model for each asset. Based on performance on validation set, decide whether to trade or not.
    Allocation uniform across traded coins.
    """
    def __init__(self,data,coin_names,thres_val=.4,max_gap=.4):
        """
        Init with: --- data: a dictionary containing training and validation sets for each of the coins.
                   --- thres_val: worst allowed performance on validation set for a model on a particular coin to be trading.
                   --- max_gap: worst allowed difference between training and validation error.
        """
        models = {}
        for coin in data:
            print("Training coin: " + coin_names[coin])
            v = data[coin]
            dtrain = v["train"]
            X_train = dtrain[0]
            y_train = dtrain[1]
            dvalid = v["valid"]
            X_valid = dvalid[0]
            y_valid = dvalid[1]
            model = BoostedTreeModel()
            model.train(X_train,y_train)
            model.validate(X_valid,y_valid)
            models[coin_names[coin]] = model
        self.models = models
        self.select_traders(thres_val,max_gap)
    def select_traders(self,thres_val=.4,max_gap=.4):
        traders = {}
        coins = []
        for coin in self.models:
            model = self.models[coin]
            if model.valid_error <= thres_val:
                if model.valid_error - model.train_error <= max_gap:
                    traders[coin] = model
                    coins.append(coin)
        self.coins = coins
        self.traders = traders
    def forward(self,obs,prev_w):
        pass


class BoostedTreeModel:
    def __init__(self,params=DEFAULT_PARAMS,num_rounds=10):
        self.params = params
        self.num_rounds = num_rounds
    def train(self,X,y):
        dtrain = xgb.DMatrix(X,label=y)
        self.bst = xgb.train(self.params, dtrain, self.num_rounds)
        self.train_error = float(self.bst.eval(dtrain)[15:20])
        if self.params["silent"] == 0:
            print("Training error")
            print(self.train_error)
    def validate(self,X,y):
        dvalid = xgb.DMatrix(X,label=y)
        self.valid_error = float(self.bst.eval(dvalid)[15:20])
        if self.params["silent"] == 0:
            print("Validation error")
            print(self.valid_error)
    def predict(self,X):
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)


def data_maker(history,lookback_window=10, style="classification", shuffle=False, split_train=.7,normalization=None):
    """
    Given a slice of market history, outputs training and validation sets.
    History: nd numpy array of shape [NUM_FEATURES, NUM_ASSETS, T]
    Makes labels according to objective parameter:
        -- regression attempts to predict actual prices.
        -- classification attempts to predict direction of next return 0: negative; 1: positive
    Normalization scheme:
        -- None: use straight prices.
        -- "returns": compute returns (Pt-Pt-1)/Pt-Pt-1
        -- "divide": Pt/Po
    Lookback_window: number of previous timesteps to use at each point.
    Split: tuple containing relative size as a percentage of training, validation and test sets respectively.

    Returns a dictionary with:
            Key: coin index
            Value: a dictionary with:
                    Keys: "train" and "valid"
                    Value: Tuple (X,y) where X is numpy array of shape [set_fraction * T, NUM_FEATURES * lookback_window]
    """
    data = {}
    NUM_FEATURES, NUM_ASSETS, T = history.shape
    for coin in range(NUM_ASSETS):
        prices = history[:,coin,:]
        returns = np.diff(prices,axis=1) / prices[:,:-1]
        if normalization == "returns":
            series = returns
            offset = 0
        elif normalization == "divide":
            Po = prices[:,0].reshape(-1,1)
            series = prices / Po
            offset = 1
        else:
            series = prices
            offset = 1
        X = []
        y = []
        for i in range(lookback_window,len(series[0])):
            X.append(series[:,i-lookback_window:i].reshape(-1))
            ret = returns[CLOSE_INDEX,i-offset]
            if style == "classification":
                y.append((np.sign(ret)+1)/2)
            elif style == "regression":
                y.append(ret)
        if shuffle is True:
            X,y = shuffle_(X,y)
        else:
            X,y = np.array(X),np.array(y)
        s0 = int(split_train*T)
        X_train, y_train = X[:s0], y[:s0]
        X_valid, y_valid = X[s0:], y[s0:]
        data[coin] = {"train":(X_train,y_train),"valid":(X_valid,y_valid)}
    return data

def shuffle_(x,y):
    mix = list(zip(x,y))
    np.random.shuffle(mix)
    res = list(zip(*mix))
    x = res[0]
    y = res[1]
    return np.array(x),np.array(y)
