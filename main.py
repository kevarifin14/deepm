import json

from markethistory import MarketHistory
from constants import * 
from agent import Agent
from networks import *

def get_data(config):
    data_global = MarketHistory(config).data
    num_feature, num_asset, T = data_global.shape
    btc_price_tensor = np.ones((num_feature, 1, T))
    data_global = np.concatenate((btc_price_tensor, data_global), axis=1)
    print("Global data tensor shape:", data_global.shape)
    return data_global

with open('config.json') as file:
    config = json.load(file)

policy = DecisionNetwork_CNN()
data = get_data(config)
agent = Agent(policy, config, data=data)

agent.train()

