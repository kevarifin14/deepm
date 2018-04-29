from markethistory import MarketHistory
from constants import * 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Backtest:
    
    '''Performs basic functionalities of a trading simulation.
       Parameters: --- start and end dates for training and testing, as string of format year/month/day
                   --- agent, whether trained or not, with functionality get_policy(data) which generates a numpy matrix
                   of rank 2 and dimensions (n_timesteps, n_coins) containing the portfolio allocation at each time step.
                   Note 1: each row in that matrix should have L1 norm equal to 1 (sum of absolute values), otherwise, leverage is  
                   being used.
                   Note 2: if agent is None, generate random long only portfolio allocations.
       Usage: automatically computes performance summary on both training and test sets. User can call plot_performance() to see
       portfolio value with respect to time (geometric returns).
       TODOS: -- Add visualization functionalities for trading performance such as max drawdown, # of trades per asset, 
       performance per asset etc.
              -- Enable user to automatically save the backtest and agent to some location.'''
    
    def __init__(self,agent,start_training,end_training,start_testing,end_testing,period=1800,include_fees=True):
        self.agent = agent
        self.period = period
        self.start_training_date = start_training
        self.end_training_date = end_training
        self.start_testing_date = start_testing
        self.end_testing_date = end_testing
        mh_train = MarketHistory(start_training,end_training)
        self.coins = mh_train.traded_coins
        self.data_train = mh_train.data
        self.data_test = MarketHistory(start_testing,end_testing).data
        if agent is None:
            policy_train,policy_test = None,None
        else:
            policy_train = agent.get_policy(self.data_train)
            policy_test = agent.get_policy(self.data_test)
        self.train_returns = self.calculate_returns(self.data_train,policy_train,include_fees=include_fees)
        self.test_returns = self.calculate_returns(self.data_test,policy_test,include_fees=include_fees)
        self.performance_summary()
        
    def calculate_returns(self,data,policy=None,include_fees=True):
        if policy is None:
            weights = np.random.rand(data.shape[2]-1,data.shape[1])
            weights = weights / np.sum(weights,axis=1).reshape(-1,1)
        else:
            weights = policy[:-1,:]
        prices_trading = data[0,:,:]
        returns = np.diff(prices_trading,axis=1)
        ret = (returns / prices_trading[:,:-1]).T
        return_per_stamp = np.sum(weights * ret,axis=1)
        count_trades = np.sum(abs(np.diff(weights,axis=0)) > 0, axis=1)
        self.total_trades = np.sum(count_trades)
        trades_per_asset = np.sum(abs(np.diff(weights,axis=0)) > 0, axis=0)
        trades = {}
        for i in range(len(trades_per_asset)):
            trades[self.coins[i]] = trades_per_asset[i]
        self.num_trades = trades
        if include_fees is True:
            count_trades = np.hstack([[0],count_trades]) #always no trade at the first timestep (all cash)
            fees = count_trades * FLAT_FEE
            fees = np.sum(weights != 0,axis=1) * FLAT_FEE
            return_per_stamp = return_per_stamp - fees
        return return_per_stamp
    
    def performance_summary(self):
        geo_returns_train = np.cumprod(self.train_returns + 1)
        geo_returns_test = np.cumprod(self.test_returns + 1)
        mean_return_train = np.mean(self.train_returns)
        mean_return_test = np.mean(self.test_returns)
        annualization_multiplier = (252 * 24 * 3600) / self.period
        annualized_return_train = (mean_return_train + 1)**annualization_multiplier
        annualized_return_test = (mean_return_test + 1)**annualization_multiplier
        sd_train = np.std(self.train_returns)
        sd_test = np.std(self.test_returns)
        sharpe_test = mean_return_test / sd_test
        sharpe_train = mean_return_train / sd_train
        sharpe_test_ann = sharpe_test * np.sqrt(annualization_multiplier)
        sharpe_train_ann = sharpe_train * np.sqrt(annualization_multiplier)
        summary = {"Train":{"Start":self.start_training_date,"End":self.end_training_date,"Average return":mean_return_train,
                           "Final portfolio value": geo_returns_train[-1],"Sharpe":sharpe_train}\
                   ,"Test":{"Start":self.start_testing_date,"End":self.end_testing_date,"Average return":mean_return_test,
                           "Final portfolio value": geo_returns_test[-1],"Sharpe":sharpe_test}}
        summary = pd.DataFrame(summary).T
        summary.columns = ["Start","End","Final portfolio value","Average return","Sharpe"]
        self.summary = summary
        print(self.summary)
        print("Number of trades per asset :")
        print(self.num_trades)
        
    def plot_performance(self):
        geo_returns_train = np.cumprod(self.train_returns + 1)
        geo_returns_test = np.cumprod(self.test_returns + 1)
        plt.plot(geo_returns_train)
        plt.title("Training portfolio performance - from " + self.start_training_date + " to " + self.end_training_date)
        plt.figure()
        plt.plot(geo_returns_test)
        plt.title("Testing portfolio performance - from " + self.start_testing_date + " to " + self.end_testing_date)
        plt.show()

def main():
    start_training = '2017/04/10'
    end_training = '2017/04/11'
    start_testing = '2017/04/12'
    end_testing = '2017/04/13'
    agent = None
    bt = Backtest(agent,start_training,end_training,start_testing,end_testing)
    bt.plot_performance()