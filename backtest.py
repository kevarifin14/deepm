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
    
    def __init__(self, agent, config):
        self.config = config['backtest']
        self.agent = agent
        self.period = self.config['period']
        self.bt_start_date = self.config['bt_start_date']
        self.bt_end_date = self.config['bt_end_date']
        self.coins = self.config['coins']
        self.flat_fee = self.config['flat_fee']

        self.data_global = self.agent.data_global        
        self.data_train = self.agent.data_train
        self.data_test = self.agent.data_test

        policy_train = self.agent.get_policy(self.data_train)
        policy_test = self.agent.get_policy(self.data_test)

        if type(self.data_train) == type({}): ##Then we know its a supervised agent with slighlty different API
            prices_train = agent.prices_train
            prices_test = agent.prices_test
        else:
            prices_train = self.data_train[0,:,:].numpy()
            prices_test = self.data_test[0,:,:].numpy()

        print('Getting backtest results for train data...')
        self.train_returns = self.calculate_returns(prices_train, policy_train, include_fees=self.config['include_fees'])
        print('Getting backtest results for test data...')
        self.test_returns = self.calculate_returns(prices_test, policy_test, include_fees=self.config['include_fees'])
        self.performance_summary()
        
    def calculate_returns(self, prices_trading, policy=None, include_fees=True):
        weights = policy[:-1, :]
        unif_alloc = 1/np.max(weights)
        returns = np.diff(prices_trading, axis=1)
        ret = (returns / prices_trading[:, :-1]).T
        return_per_stamp = np.sum(weights * ret, axis=1)
        count_trades = np.sum(abs(np.diff(weights, axis=0)) > 0, axis=1)
        self.total_trades = np.sum(count_trades)
        trades_per_asset = np.sum(abs(np.diff(weights, axis=0)) > 0, axis=0)
        trades = {}
        for i in range(len(trades_per_asset)):
            trades[self.coins[i]] = trades_per_asset[i]
        self.num_trades = trades
        if include_fees is True:
            print("include_fees")
            count_trades = np.hstack([0,count_trades]) #always no trade at the first timestep (all cash)
            fees = count_trades * self.flat_fee / unif_alloc
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
        summary = {
            "Train": {
                # "Start": self.bt_start_date,
                # "End": self.bt_end_date,
                "Average return":mean_return_train,
                "Final portfolio value": geo_returns_train[-1],
                "Sharpe":sharpe_train
            },
            "Test": {
                # "Start": self.start_testing_date,
                # "End": self.end_testing_date,
                "Average return": mean_return_test,
                "Final portfolio value": geo_returns_test[-1],
                "Sharpe": sharpe_test
            }
        }
        summary = pd.DataFrame(summary).T
        summary.columns = ['Final portfolio value', 'Average return', 'Sharpe']
        self.summary = summary
        print(self.summary)
        print('Number of trades per asset: ')
        print(self.num_trades)
        
    def plot_performance(self):
        geo_returns_train = np.cumprod(self.train_returns + 1)
        geo_returns_test = np.cumprod(self.test_returns + 1)
        plt.plot(geo_returns_train)
        # plt.title("Training portfolio performance - from " + self.bt_start_date + " to " + self.end_training_date)
        plt.figure()
        plt.plot(geo_returns_test)
        # plt.title("Testing portfolio performance - from " + self.start_testing_date + " to " + self.end_testing_date)
        plt.show()