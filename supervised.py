from markethistory import MarketHistory
from constants import * 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import json

with open('config.json') as file:
    CONFIG = json.load(file)

CLOSE_INDEX = 0 ###USED TO CALCULATE RETURNS###
DEFAULT_PARAMS = {'max_depth':5, 'eta':.25, 'silent':1,'alpha':1,'min_child_weight':5, 'objective':"binary:logistic" }

class Agent:
    """
    Train a different model for each asset. Based on performance on validation set, decide whether to trade or not.
    Allocation uniform across traded coins.
    """
    def __init__(self,markethistory,features,lookback_window=5,normalization="returns",thres_val=.4,max_gap=.4,num_rounds=10,params=DEFAULT_PARAMS,config=CONFIG):
        """
        Init with: --- data: a dictionary containing training and validation sets for each of the coins.
                   --- thres_val: worst allowed performance on validation set for a model on a particular coin to be trading.
                   --- max_gap: worst allowed difference between training and validation error.
        """
        data,feature_list,prices_train,prices_test = data_maker(markethistory,features,lookback_window=lookback_window,normalization=normalization)
        models = {}
        errors = {}
        self.data_global = data
        data_train = {}
        data_test = {}
        for coin in data:
            v = data[coin]
            dtrain = v["train"]
            data_train[coin] = dtrain
            dtest = v["test"]
            data_test[coin] = dtest
            X_train = dtrain[0]
            y_train = dtrain[1]
            dvalid = v["valid"]
            X_valid = dvalid[0]
            y_valid = dvalid[1]
            model = BoostedTreeModel(feature_list,coin,num_rounds=num_rounds,params=params)
            model.train(X_train,y_train)
            model.validate(X_valid,y_valid)
            errors[coin] = [model.train_error,model.valid_error]
            models[coin] = model

        self.prices_train = prices_train
        self.prices_test = prices_test
        self.data_train = data_train
        self.data_test = data_test
        self.models = models
        self.select_traders(thres_val,max_gap)
        print(pd.DataFrame(errors,index=["Training error","Validation error"]))
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
    def get_policy(self, data):
        """
        Input: dictionary with a key for each coin and value: numpy array of size [NUM_FEATURES, T]
        Returns: a numpy array of shape [T, NUM_ASSET].
        Notes: For the first "OBS_WINDOW" observations, allocation is considered to be entirely cash.
               Doesn't trade if recommended allocation for a particular asset is less than a threshold CUTOFF_TRADE.
        """
        preds = []
        for coin in self.traders:
            model = self.traders[coin]
            X_test = data[coin][0]
            preds_coin = model.predict(X_test)
            preds.append(preds_coin)
        preds = np.array(preds).T
        T,num_assets_traded = preds.shape
        allocs = (2*np.round(preds) - 1)/num_assets_traded ###Uniform allocation across traded coins
        allocs_sparse = []
        all_coins = list(data.keys())
        for w in allocs:
            w_sparse = np.zeros(len(all_coins))
            for i in range(len(all_coins)):
                coin = all_coins[i]
                if coin in self.traders:
                    w_sparse[i] = w[self.coins.index(coin)]
            allocs_sparse.append(w_sparse)
        return np.array(allocs_sparse)




class BoostedTreeModel:
    def __init__(self,feature_list,asset,params=DEFAULT_PARAMS,num_rounds=10):
        self.params = params
        self.num_rounds = num_rounds
        self.features = feature_list
        self.name = asset
    def train(self,X,y):
        X,y = self.clean(X,y)
        dtrain = xgb.DMatrix(X,label=y,feature_names=self.features)
        self.bst = xgb.train(self.params, dtrain, self.num_rounds)
        self.train_error = float(self.bst.eval(dtrain)[15:20])
        if self.params["silent"] == 0:
            print("Training error")
            print(self.train_error)
    def validate(self,X,y):
        X,y = self.clean(X,y)
        dvalid = xgb.DMatrix(X,label=y,feature_names=self.features)
        self.valid_error = float(self.bst.eval(dvalid)[15:20])
        if self.params["silent"] == 0:
            print("Validation error")
            print(self.valid_error)
    def predict(self,X):
        dtest = xgb.DMatrix(X,feature_names=self.features)
        return self.bst.predict(dtest)
    def plot_importance(self):
        xgb.plot_importance(self.bst,max_num_features=10)
    def plot_trees(self,num_trees=1):
        for i in range(num_trees):
            xgb.plot_tree(self.bst,num_trees=i)
    def clean(self,X,y):
        nonzero_indices = np.where(y!=0.5)[0]
        X = X[nonzero_indices]
        y = y[nonzero_indices]
        return X,y

def data_maker(history,features,lookback_window=10, style="classification", shuffle=False, splits=(.7,.15),normalization=None):
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
    data_history = history.data
    NUM_FEATURES, NUM_ASSETS, T = data_history.shape
    feature_list = []
    feature_indices = []
    for feat in features:
        feature_indices.append(features[feat])
        for i in range(lookback_window,0,-1):
            feature_list.append(feat+" T-"+str(i))
    feature_list.append("timestep")
    if normalization == "returns":
        all_prices = data_history[0,:,:-1]
    else:
        all_prices = data_history[0,:,:]
    for coin in range(NUM_ASSETS):
        prices = data_history[feature_indices,coin,:]
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
            X.append(np.hstack([series[:,i-lookback_window:i].reshape(-1),i]))
            ret = returns[CLOSE_INDEX,i-offset]
            if style == "classification":
                s = (np.sign(ret)+1)/2.
                y.append(s)
            elif style == "regression":
                y.append(ret)
        if shuffle is True:
            X,y = shuffle_(X,y)
        else:
            X,y = np.array(X),np.array(y)
        split_train, split_valid = splits
        s0 = int(split_train*T)
        s1 = s0 + int(split_valid*T)
        X_train, y_train = X[:s0], y[:s0]
        X_valid, y_valid = X[s0:s1], y[s0:s1]
        X_test, y_test = X[s1:], y[s1:]
        prices_train = all_prices[:,lookback_window:s0+lookback_window]
        prices_test = all_prices[:,s1+lookback_window:]
        data[history.traded_coins[coin]] = {"train":(X_train,y_train),"valid":(X_valid,y_valid),"test":(X_test,y_test)}
    return data,feature_list, prices_train, prices_test

def shuffle_(x,y):
    mix = list(zip(x,y))
    np.random.shuffle(mix)
    res = list(zip(*mix))
    x = res[0]
    y = res[1]
    return np.array(x),np.array(y)
